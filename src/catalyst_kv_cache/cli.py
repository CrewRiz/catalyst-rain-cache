from __future__ import annotations

import argparse
import json
import sys

from catalyst_kv_cache.sdk_bridge import demo_payload, onboarding_payload, sdk_status
from catalyst_kv_cache.serve import CatalystServeConfig, readiness_payload, serve_forever


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if not argv or argv[0].startswith("-"):
        return _smoke(argv)
    command = argv[0]
    rest = argv[1:]
    if command == "smoke":
        return _smoke(rest)
    if command == "doctor":
        return _doctor(rest)
    if command == "onboard":
        return _onboard(rest)
    if command == "demo":
        return _demo(rest)
    if command == "serve":
        return _serve(rest)
    parser = _base_parser()
    parser.error(f"unknown command: {command}")
    return 2


def _base_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Catalyst KV Cache adapter CLI.")
    parser.add_argument("command", nargs="?", choices=("smoke", "doctor", "onboard", "demo", "serve"))
    return parser


def _smoke(argv: list[str] | None = None) -> int:
    from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig

    parser = argparse.ArgumentParser(description="Run a Catalyst KV Cache smoke test.")
    parser.add_argument("--mode", choices=("passthrough", "refs"), default="passthrough")
    args = parser.parse_args(argv)

    cache = CatalystKVCache(CatalystKVConfig(mode=args.mode, dim=1024))
    key_states = [[float(i), float(i + 1)] for i in range(32)]
    value_states = [[float(i * 2), float(i * 2 + 1)] for i in range(32)]
    result = cache.update(key_states, value_states, layer_idx=0, cache_kwargs={"position": 0})
    report = cache.compression_report()
    payload = {
        "mode": args.mode,
        "update_return_type": type(result).__name__,
        "seq_length": cache.get_seq_length(0),
        "report": report,
    }
    cache.reset()
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


def _doctor(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Catalyst KV Cache SDK readiness.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = sdk_status()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        _print_human_status(payload)
    return 0


def _onboard(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Show drop-in adapter onboarding steps.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = onboarding_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print("Catalyst KV Cache onboarding")
        print("")
        print("Python:")
        for line in payload["python_api"]:
            print(f"  {line}")
        print("")
        print("Commands:")
        for command in payload["commands"]:
            print(f"  {command}")
    return 0


def _demo(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize publishable Catalyst RAIN cache demo evidence.")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = demo_payload()
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        official = payload["official_subset"]
        longbench = official["longbench_v2"]
        ruler = official["ruler"]
        probe = payload["live_model_probe"]
        print("Catalyst RAIN cache demo")
        print(f"  Verdict: {payload['headline']['verdict']}")
        print(f"  LongBench v2 subset: {longbench['accuracy_pct']}% over {longbench['sample_count']} examples")
        print(f"  RULER subset: {ruler['mean_score_pct']}% over {ruler['prediction_count']} predictions")
        print(f"  Live model probe: {probe['status']} via {probe['auth_source']}")
        print("  Boundary: public adapter only; private algorithms live in catalyst-brain")
    return 0


def _print_human_status(payload: dict[str, object]) -> None:
    print("Catalyst KV Cache doctor")
    print(f"  SDK installed: {payload['sdk_installed']}")
    print(f"  SDK version: {payload['sdk_version']}")
    print("  Public boundary: algorithms live in catalyst-brain")
    features = payload["sdk_features"]
    if isinstance(features, dict):
        for name, available in features.items():
            print(f"  {name}: {available}")


def _serve(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Catalyst OpenAI-compatible serving shim.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8088)
    parser.add_argument("--model", default="catalyst-private-ai")
    parser.add_argument("--cache-mode", choices=("passthrough", "refs"), default="passthrough")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    config = CatalystServeConfig(
        host=args.host,
        port=args.port,
        model=args.model,
        cache_mode=args.cache_mode,
    )
    payload = readiness_payload(config)
    if args.dry_run:
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            print(f"Catalyst Serve shim ready on http://{config.host}:{config.port}")
            for route in payload["routes"]:
                print(f"  {route}")
        return 0
    print(f"Catalyst Serve shim listening on http://{config.host}:{config.port}")
    serve_forever(config)
    return 0
