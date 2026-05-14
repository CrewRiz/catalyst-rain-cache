from __future__ import annotations

import json
import threading
import urllib.request


def test_chat_completion_response_matches_openai_shape():
    from catalyst_kv_cache.serve import chat_completion_response

    payload = {
        "model": "local-private-model",
        "messages": [{"role": "user", "content": "hello"}],
    }

    response = chat_completion_response(payload)

    assert response["object"] == "chat.completion"
    assert response["model"] == "local-private-model"
    assert response["choices"][0]["message"]["role"] == "assistant"
    assert "catalyst" in response
    assert response["usage"]["total_tokens"] >= response["usage"]["prompt_tokens"]


def test_serving_http_routes_return_openai_compatible_json():
    from catalyst_kv_cache.serve import CatalystServeConfig, make_server

    server = make_server(CatalystServeConfig(port=0))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = server.server_address[:2]
        base = f"http://{host}:{port}"

        with urllib.request.urlopen(f"{base}/healthz", timeout=5) as response:
            health = json.loads(response.read().decode("utf-8"))
        assert health["status"] == "ok"
        assert health["public_boundary"]["ships_private_algorithms"] is False

        request = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps({"model": "catalyst-test", "messages": [{"role": "user", "content": "ping"}]}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            completion = json.loads(response.read().decode("utf-8"))
        assert completion["object"] == "chat.completion"
        assert completion["choices"][0]["finish_reason"] == "stop"

        with urllib.request.urlopen(f"{base}/v1/models", timeout=5) as response:
            models = json.loads(response.read().decode("utf-8"))
        assert models["object"] == "list"
        assert models["data"][0]["id"].startswith("catalyst")
    finally:
        server.shutdown()
        server.server_close()


def test_cli_serve_dry_run_reports_openai_routes(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["serve", "--dry-run", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["server"] == "catalyst-serve-shim"
    assert "/v1/chat/completions" in payload["routes"]
    assert payload["public_boundary"]["algorithms_live_in"] == "catalyst-brain"
