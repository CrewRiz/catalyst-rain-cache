from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from catalyst_kv_cache.sdk_bridge import sdk_status


@dataclass(frozen=True)
class CatalystServeConfig:
    host: str = "127.0.0.1"
    port: int = 8088
    model: str = "catalyst-private-ai"
    cache_mode: str = "passthrough"


ROUTES = ("/healthz", "/v1/models", "/v1/chat/completions")


def readiness_payload(config: CatalystServeConfig | None = None) -> dict[str, Any]:
    config = config or CatalystServeConfig()
    status = sdk_status()
    return {
        "server": "catalyst-serve-shim",
        "status": "ok",
        "host": config.host,
        "port": config.port,
        "model": config.model,
        "cache_mode": config.cache_mode,
        "routes": list(ROUTES),
        "sdk": {
            "package": status["sdk_package"],
            "installed": status["sdk_installed"],
            "version": status["sdk_version"],
            "features": status["sdk_features"],
        },
        "public_boundary": status["public_boundary"],
    }


def models_response(config: CatalystServeConfig | None = None) -> dict[str, Any]:
    config = config or CatalystServeConfig()
    return {
        "object": "list",
        "data": [
            {
                "id": config.model,
                "object": "model",
                "created": 0,
                "owned_by": "catalyst-brain",
            }
        ],
    }


def chat_completion_response(
    payload: dict[str, Any],
    *,
    config: CatalystServeConfig | None = None,
) -> dict[str, Any]:
    config = config or CatalystServeConfig()
    model = str(payload.get("model") or config.model)
    messages = payload.get("messages") or []
    prompt_tokens = _approx_message_tokens(messages)
    content = (
        "Catalyst Serve shim is ready. Attach a private model backend through "
        "catalyst-brain to generate model text while keeping HKVC internals "
        "behind the SDK boundary."
    )
    completion_tokens = _approx_tokens(content)
    return {
        "id": f"chatcmpl-catalyst-{uuid.uuid4().hex[:16]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content,
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "catalyst": {
            "adapter": "catalyst-kv-cache",
            "server": "catalyst-serve-shim",
            "cache_mode": config.cache_mode,
            "sdk_ready": any(bool(value) for value in sdk_status()["sdk_features"].values()),
            "ships_private_algorithms": False,
        },
    }


def make_server(config: CatalystServeConfig | None = None) -> ThreadingHTTPServer:
    config = config or CatalystServeConfig()

    class CatalystServeHandler(BaseHTTPRequestHandler):
        server_version = "CatalystServeShim/0.1"

        def do_GET(self) -> None:  # noqa: N802
            if self.path == "/healthz":
                self._send_json(readiness_payload(config))
                return
            if self.path == "/v1/models":
                self._send_json(models_response(config))
                return
            self._send_json({"error": {"message": "not found", "type": "not_found"}}, status=404)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != "/v1/chat/completions":
                self._send_json({"error": {"message": "not found", "type": "not_found"}}, status=404)
                return
            try:
                payload = self._read_json()
            except ValueError as exc:
                self._send_json({"error": {"message": str(exc), "type": "invalid_request_error"}}, status=400)
                return
            self._send_json(chat_completion_response(payload, config=config))

        def log_message(self, *_args: Any) -> None:
            return

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length).decode("utf-8") if length else "{}"
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError("request body must be a JSON object")
            return value

        def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
            body = json.dumps(payload, sort_keys=True).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return ThreadingHTTPServer((config.host, config.port), CatalystServeHandler)


def serve_forever(config: CatalystServeConfig | None = None) -> None:
    server = make_server(config)
    try:
        server.serve_forever()
    finally:
        server.server_close()


def _approx_message_tokens(messages: Any) -> int:
    if not isinstance(messages, list):
        return 0
    return sum(_approx_tokens(str(item.get("content", ""))) for item in messages if isinstance(item, dict))


def _approx_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0
