from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig  # noqa: E402


OFFICIAL_BENCHMARKS = [
    {
        "name": "LongBench",
        "source_url": "https://github.com/THUDM/LongBench",
        "dataset_id": "THUDM/LongBench-v2",
        "runner_status": "not_run",
        "next_step": "Run official LongBench/LongBench-v2 tasks against a model/API path with recorded dataset revision.",
    },
    {
        "name": "RULER",
        "source_url": "https://github.com/NVIDIA/RULER",
        "dataset_id": "NVIDIA/RULER",
        "runner_status": "not_run",
        "next_step": "Run NVIDIA RULER configs at target context lengths and record exact task config.",
    },
]


BASELINES = [
    {
        "method": "KIVI",
        "import_name": "kivi",
        "comparison_role": "quantized_kv_cache",
        "next_step": "Import an implementation or pinned fork and run the same tensor/model workload.",
    },
    {
        "method": "TurboQuant",
        "import_name": "turboquant",
        "comparison_role": "quantized_kv_cache",
        "next_step": "Import implementation artifacts or reproduce from paper code if released.",
    },
    {
        "method": "PyramidKV/H2O retention",
        "import_name": "pyramidkv",
        "comparison_role": "eviction_or_retention",
        "next_step": "Run retention baselines under identical context and scoring prompts.",
    },
    {
        "method": "vLLM/PagedAttention",
        "import_name": "vllm",
        "comparison_role": "serving_memory_manager",
        "next_step": "Run vLLM on the same host and capture KV residency plus throughput.",
    },
    {
        "method": "Sparse/Retrieval Attention",
        "import_name": "faiss",
        "comparison_role": "retrieval_or_sparse_attention",
        "next_step": "Run a sparse/retrieval attention implementation on the same fixed corpus.",
    },
]


CF_WORKERS_AI_DOCS = "https://developers.cloudflare.com/workers-ai/models/"
CF_DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"


def run_next_evidence(
    *,
    output: str | Path,
    chart_dir: str | Path,
    updates: int = 512,
    network: bool = True,
) -> dict[str, Any]:
    official = _official_benchmark_readiness(network=network)
    adapter = _public_adapter_latency(updates=updates)
    baselines = _baseline_import_readiness()
    cloudflare = _cloudflare_workers_ai_probe(network=network)
    payload = {
        "system_info": {
            "suite": "next_evidence_tier",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "updates": updates,
        },
        "claim_boundaries": {
            "official_scores_not_yet_claimed": True,
            "baseline_rows_are_import_readiness_until_measured": True,
            "public_adapter_latency_is_not_sdk_kernel_latency": True,
            "cloudflare_workers_ai_optional_provider": True,
            "private_algorithms_remain_in": "catalyst_brain",
        },
        "official_benchmark_readiness": official,
        "public_adapter_latency": adapter,
        "baseline_import_readiness": baselines,
        "cloudflare_workers_ai": cloudflare,
        "remaining_blockers": _remaining_blockers(official, baselines, cloudflare),
        "charts": {
            "next_official_readiness": "charts/next_official_readiness.svg",
            "next_adapter_latency": "charts/next_adapter_latency.svg",
            "next_baseline_imports": "charts/next_baseline_imports.svg",
            "next_cloudflare_ai": "charts/next_cloudflare_ai.svg",
        },
    }
    output_path = Path(output)
    chart_path = Path(chart_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.mkdir(parents=True, exist_ok=True)
    _write_charts(payload, chart_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _official_benchmark_readiness(*, network: bool) -> dict[str, Any]:
    rows = []
    for row in OFFICIAL_BENCHMARKS:
        source = dict(row)
        reachable = _url_reachable(source["source_url"]) if network else None
        source["source_reachable"] = reachable
        source["readiness_status"] = "source_reachable_runner_not_executed" if reachable else "runner_not_executed"
        rows.append(source)
    return {
        "aggregate": {
            "tracked_benchmark_count": len(rows),
            "source_reachable_count": sum(1 for row in rows if row["source_reachable"] is True),
            "official_score_count": 0,
        },
        "rows": rows,
    }


def _public_adapter_latency(*, updates: int) -> dict[str, Any]:
    passthrough = _measure_adapter_mode(mode="passthrough", updates=updates)
    refs = _measure_adapter_mode(mode="refs", updates=updates)
    return {
        "aggregate": {
            "updates": updates,
            "passthrough_update_p50_us": passthrough["p50_us"],
            "refs_update_p50_us": refs["p50_us"],
            "passthrough_update_p95_us": passthrough["p95_us"],
            "refs_update_p95_us": refs["p95_us"],
            "refs_compact_saved_pct": refs["compression_report"]["saved_pct"],
            "refs_total_compact_bytes": refs["compression_report"]["total_compact_bytes"],
        },
        "rows": [passthrough, refs],
    }


def _measure_adapter_mode(*, mode: str, updates: int) -> dict[str, Any]:
    cache = CatalystKVCache(CatalystKVConfig(dim=256, mode=mode, purpose="research"))
    timings = []
    returned_refs = 0
    returned_passthrough = 0
    for index in range(updates):
        key = _vector(index, 16)
        value = _vector(index + 10_000, 16)
        started = time.perf_counter_ns()
        result = cache.update(
            key,
            value,
            layer_idx=index % 4,
            cache_kwargs={"position": index, "token_count": 1},
        )
        timings.append((time.perf_counter_ns() - started) / 1000.0)
        if isinstance(result, dict) and "kv_ref" in result:
            returned_refs += 1
        elif isinstance(result, tuple):
            returned_passthrough += 1
    return {
        "mode": mode,
        "updates": updates,
        "p50_us": round(statistics.median(timings), 4),
        "p95_us": round(_percentile(timings, 0.95), 4),
        "mean_us": round(statistics.fmean(timings), 4),
        "returned_refs": returned_refs,
        "returned_passthrough": returned_passthrough,
        "compression_report": cache.compression_report(),
    }


def _baseline_import_readiness() -> dict[str, Any]:
    rows = []
    for row in BASELINES:
        package_found = importlib.util.find_spec(row["import_name"]) is not None
        rows.append(
            {
                **row,
                "package_found": package_found,
                "readiness_status": "importable_not_run" if package_found else "package_not_installed",
                "measured_same_hardware": False,
            }
        )
    return {
        "aggregate": {
            "tracked_baseline_count": len(rows),
            "importable_count": sum(1 for row in rows if row["package_found"]),
            "measured_same_hardware_count": 0,
        },
        "rows": rows,
    }


def _cloudflare_workers_ai_probe(*, network: bool) -> dict[str, Any]:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_AUTH_TOKEN")
    model = os.environ.get("CATALYST_CF_WORKERS_AI_MODEL", CF_DEFAULT_MODEL)
    configured = bool(network and account_id and api_token)
    status = "configured" if configured else "skipped_missing_credentials"
    if account_id and api_token and not network:
        status = "skipped_network_disabled"
    payload: dict[str, Any] = {
        "provider": "Cloudflare Workers AI",
        "docs_url": CF_WORKERS_AI_DOCS,
        "model": model,
        "account_id_present": bool(account_id),
        "api_token_present": bool(api_token),
        "status": status,
        "quality_probe": {
            "ran": False,
            "score": None,
            "latency_ms": None,
            "prompt_count": 0,
        },
        "setup": {
            "required_env": ["CLOUDFLARE_ACCOUNT_ID", "CLOUDFLARE_API_TOKEN"],
            "optional_env": ["CATALYST_CF_WORKERS_AI_MODEL"],
            "default_model": CF_DEFAULT_MODEL,
        },
    }
    if not configured:
        return payload
    try:
        started = time.perf_counter_ns()
        rows = [_cloudflare_prompt(account_id=account_id, api_token=api_token, model=model)]
        latency_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        score = statistics.fmean(row["correct"] for row in rows)
        payload["status"] = "measured"
        payload["quality_probe"] = {
            "ran": True,
            "score": round(score, 4),
            "latency_ms": round(latency_ms, 4),
            "prompt_count": len(rows),
            "rows": rows,
        }
    except Exception as exc:
        payload["status"] = "error"
        payload["error"] = repr(exc)
    return payload


def _cloudflare_prompt(*, account_id: str, api_token: str, model: str) -> dict[str, Any]:
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run/{model}"
    body = {
        "messages": [
            {
                "role": "system",
                "content": "Answer with only the requested fact. No explanation.",
            },
            {
                "role": "user",
                "content": (
                    "Context: The Northstar contract renewal notice code is ALPHA-17.\n"
                    "Question: What is the renewal notice code?"
                ),
            },
        ],
        "max_tokens": 32,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        decoded = json.loads(response.read().decode("utf-8"))
    text = _cloudflare_response_text(decoded)
    return {
        "id": "northstar_code",
        "expected": "ALPHA-17",
        "response": text,
        "correct": "ALPHA-17" in text,
    }


def _cloudflare_response_text(decoded: dict[str, Any]) -> str:
    result = decoded.get("result", decoded)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "text", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    return json.dumps(decoded, sort_keys=True)[:500]


def _remaining_blockers(
    official: dict[str, Any],
    baselines: dict[str, Any],
    cloudflare: dict[str, Any],
) -> list[str]:
    blockers = [
        "Official LongBench/RULER scores still require running official task repos with pinned dataset revisions.",
        "KIVI, TurboQuant, PyramidKV/H2O, vLLM/PagedAttention, and sparse/retrieval baselines still need imported implementation runs on the same hardware.",
        "Public adapter timings measure wrapper overhead, not production catalyst-brain SDK kernel throughput.",
    ]
    if cloudflare["status"] == "skipped_missing_credentials":
        blockers.append("Cloudflare Workers AI stronger-model probe requires CLOUDFLARE_API_TOKEN before it can run.")
    if cloudflare["status"] == "skipped_network_disabled":
        blockers.append("Cloudflare Workers AI stronger-model probe was skipped because network execution was disabled.")
    if official["aggregate"]["official_score_count"] == 0:
        blockers.append("No official benchmark score is claimed in this artifact.")
    if baselines["aggregate"]["measured_same_hardware_count"] == 0:
        blockers.append("No modern baseline implementation has been measured on the same machine yet.")
    return blockers


def _url_reachable(url: str) -> bool:
    request = urllib.request.Request(url, method="GET", headers={"User-Agent": "catalyst-kv-cache-evidence"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return 200 <= int(response.status) < 400
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _vector(seed: int, dim: int) -> list[float]:
    return [((seed * 31 + index * 17) % 101) / 100.0 for index in range(dim)]


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * quantile)))
    return ordered[index] if ordered else 0.0


def _write_charts(payload: dict[str, Any], chart_dir: Path) -> None:
    official = payload["official_benchmark_readiness"]
    adapter = payload["public_adapter_latency"]
    baselines = payload["baseline_import_readiness"]
    cloudflare = payload["cloudflare_workers_ai"]
    (chart_dir / "next_official_readiness.svg").write_text(
        _bar_chart(
            "Official Benchmark Readiness",
            [
                ("Tracked", official["aggregate"]["tracked_benchmark_count"], "#0f766e"),
                ("Reachable", official["aggregate"]["source_reachable_count"], "#2563eb"),
                ("Scores", official["aggregate"]["official_score_count"], "#b45309"),
            ],
        )
    )
    (chart_dir / "next_adapter_latency.svg").write_text(
        _bar_chart(
            "Public Adapter Update Latency",
            [
                ("Passthrough p50", adapter["aggregate"]["passthrough_update_p50_us"], "#0f766e"),
                ("Refs p50", adapter["aggregate"]["refs_update_p50_us"], "#2563eb"),
                ("Refs p95", adapter["aggregate"]["refs_update_p95_us"], "#7c2d12"),
            ],
        )
    )
    (chart_dir / "next_baseline_imports.svg").write_text(
        _bar_chart(
            "Baseline Import Readiness",
            [
                ("Tracked", baselines["aggregate"]["tracked_baseline_count"], "#334155"),
                ("Importable", baselines["aggregate"]["importable_count"], "#0f766e"),
                ("Measured", baselines["aggregate"]["measured_same_hardware_count"], "#b45309"),
            ],
        )
    )
    cf_value = 1.0 if cloudflare["status"] == "measured" else 0.0
    token_value = 1.0 if cloudflare["api_token_present"] else 0.0
    (chart_dir / "next_cloudflare_ai.svg").write_text(
        _bar_chart(
            "Cloudflare Workers AI Probe",
            [
                ("Account", 1.0 if cloudflare["account_id_present"] else 0.0, "#0f766e"),
                ("Token", token_value, "#2563eb"),
                ("Measured", cf_value, "#b45309"),
            ],
            max_value=1.0,
        )
    )


def _bar_chart(title: str, values: list[tuple[str, float, str]], *, max_value: float | None = None) -> str:
    max_seen = max_value if max_value is not None else max(float(value) for _label, value, _color in values)
    body = []
    for index, (label, value, color) in enumerate(values):
        x = 90 + index * 170
        height = max(3.0, 235.0 * float(value) / max(max_seen, 1e-12))
        body.append(f'<rect x="{x}" y="{330 - height:.1f}" width="82" height="{height:.1f}" rx="5" fill="{color}"/>')
        body.append(f'<text x="{x + 41}" y="{max(20, 320 - height):.1f}" text-anchor="middle" font-size="11" fill="#111827">{float(value):.2f}</text>')
        body.append(f'<text x="{x + 41}" y="364" text-anchor="middle" font-size="11" fill="#111827">{label}</text>')
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="760" height="430" viewBox="0 0 760 430">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f'<text x="56" y="44" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>'
        '<text x="56" y="70" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">Generated by the public catalyst-kv-cache evidence runner.</text>'
        '<line x1="70" y1="330" x2="690" y2="330" stroke="#d1d5db"/>'
        f'{"".join(body)}</svg>\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the next public HKVC evidence tier.")
    parser.add_argument("--output", default="site/next_evidence_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()
    payload = run_next_evidence(
        output=args.output,
        chart_dir=args.chart_dir,
        updates=args.updates,
        network=not args.no_network,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
