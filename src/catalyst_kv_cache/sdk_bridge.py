from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path
from typing import Any


SDK_FEATURES = (
    "CatalystDynamicKVCache",
    "CatalystHolographicKVCache",
    "CatalystKVCacheReplacement",
    "CatalystLosslessAttentionReplacement",
    "run_hkvc_benchmarks",
    "run_hkvc_lossless_equivalence_benchmark",
    "run_hkvc_lossless_scale_benchmark",
    "run_hkvc_tier2_evidence",
    "run_hkvc_attention_benchmark",
    "run_hkvc_private_context_benchmark",
)


def load_catalyst_brain() -> Any | None:
    try:
        return import_module("catalyst_brain")
    except Exception:
        return None


def sdk_status() -> dict[str, Any]:
    sdk = load_catalyst_brain()
    version = getattr(sdk, "__version__", None) if sdk is not None else None
    features = {name: bool(sdk is not None and hasattr(sdk, name)) for name in SDK_FEATURES}
    return {
        "adapter_package": "catalyst-kv-cache",
        "sdk_package": "catalyst-brain",
        "sdk_installed": sdk is not None,
        "sdk_version": version,
        "sdk_features": features,
        "public_boundary": {
            "algorithms_live_in": "catalyst-brain",
            "ships_private_algorithms": False,
            "adapter_role": "integration, onboarding, checked artifacts, and evidence wrappers",
        },
    }


def create_transformers_cache(
    *,
    dim: int = 4096,
    mode: str = "passthrough",
    purpose: str = "research",
    max_tokens: int = 1024,
    prefer_sdk: bool = True,
) -> Any:
    """Create the best available HF-style cache without exposing SDK internals."""
    if prefer_sdk:
        sdk = load_catalyst_brain()
        if sdk is not None:
            if mode in {"passthrough", "dynamic"} and hasattr(sdk, "CatalystDynamicKVCache"):
                try:
                    return sdk.CatalystDynamicKVCache(dim=dim)
                except Exception:
                    pass
            if mode in {"refs", "holographic", "bounded"} and hasattr(sdk, "CatalystHolographicKVCache"):
                try:
                    return sdk.CatalystHolographicKVCache(max_tokens=max_tokens, dim=dim)
                except Exception:
                    pass
            if hasattr(sdk, "CatalystKVCacheReplacement"):
                try:
                    return sdk.CatalystKVCacheReplacement(max_tokens=max_tokens, dim=dim)
                except Exception:
                    pass

    from catalyst_kv_cache.core import CatalystKVCache, CatalystKVConfig

    fallback_mode = "refs" if mode in {"refs", "holographic", "bounded"} else "passthrough"
    return CatalystKVCache(CatalystKVConfig(dim=dim, mode=fallback_mode, purpose=purpose))


def onboarding_payload() -> dict[str, Any]:
    return {
        "goal": "drop_in_long_context_private_ai",
        "python_api": [
            "create_transformers_cache",
            "from catalyst_kv_cache import create_transformers_cache",
            "cache = create_transformers_cache(mode='passthrough')",
        ],
        "commands": [
            "catalyst-kv-cache doctor",
            "catalyst-kv-cache smoke --mode passthrough",
            "catalyst-kv-cache serve --dry-run",
            "python -m bench.lossless_equivalence --output site/lossless_equivalence_results.json --chart-dir site/charts",
            "python -m bench.lossless_scale --output site/lossless_scale_results.json --chart-dir site/charts",
            "catalyst-kv-cache onboard",
        ],
        "next_evidence": [
            "Install a catalyst-brain build with HKVC evidence exports.",
            "Run the lossless equivalence wrapper against the private SDK.",
            "Run the scaled lossless wrapper before broadening public claims.",
            "Run public evidence wrappers to regenerate charts.",
            "Move production latency work into SDK kernels, not the public adapter.",
        ],
    }


def demo_payload(*, evidence_root: str | Path | None = None) -> dict[str, Any]:
    root = Path(evidence_root) if evidence_root is not None else _default_evidence_root()
    breakthrough = _read_json(root / "site" / "breakthrough_assessment.json")
    official = _read_json(root / "site" / "official_longbench_ruler_results.json")
    next_evidence = _read_json(root / "site" / "next_evidence_results.json")
    transport = _read_json(root / "site" / "rain_transport_probe_results.json")

    longbench = official.get("longbench_v2", {}).get("aggregate", {}) if official else {}
    ruler = official.get("ruler", {}).get("aggregate", {}) if official else {}
    cloudflare = next_evidence.get("cloudflare_workers_ai", {}) if next_evidence else {}
    transport_aggregate = transport.get("aggregate", {}) if transport else {}
    blockers = list(official.get("remaining_blockers", [])) if official else []
    blockers.extend(next_evidence.get("remaining_blockers", []) if next_evidence else [])

    return {
        "demo_ready": bool(breakthrough and official and next_evidence and transport),
        "public_boundary": {
            "algorithms_live_in": "catalyst-brain",
            "ships_private_algorithms": False,
            "adapter_role": "integration, onboarding, checked artifacts, and evidence wrappers",
        },
        "headline": {
            "verdict": breakthrough.get("overall_verdict") if breakthrough else None,
            "scope": breakthrough.get("short_answer") if breakthrough else None,
        },
        "official_subset": {
            "model": official.get("system_info", {}).get("model") if official else None,
            "execution_path": official.get("system_info", {}).get("execution_path") if official else None,
            "longbench_v2": {
                "sample_count": longbench.get("sample_count", 0),
                "accuracy_pct": longbench.get("accuracy_pct"),
                "truncated_count": longbench.get("truncated_count", 0),
            },
            "ruler": {
                "prediction_count": ruler.get("prediction_count", 0),
                "mean_score_pct": ruler.get("mean_score_pct"),
                "min_score_pct": ruler.get("min_score_pct"),
            },
        },
        "live_model_probe": {
            "provider": cloudflare.get("provider"),
            "status": cloudflare.get("status"),
            "auth_source": cloudflare.get("auth_source"),
            "score": cloudflare.get("quality_probe", {}).get("score") if cloudflare else None,
            "latency_ms": cloudflare.get("quality_probe", {}).get("latency_ms") if cloudflare else None,
        },
        "rain_transport": {
            "sample_count": transport_aggregate.get("sample_count", 0),
            "mean_packed_vs_raw_json_ratio": transport_aggregate.get("mean_packed_vs_raw_json_ratio"),
        },
        "claim_boundaries": {
            "official_subset_not_full_leaderboard": True,
            "longbench_cloudflare_not_catalyst_adapter_quality": True,
            "private_algorithms_remain_in": "catalyst_brain",
        },
        "remaining_blockers": sorted(set(str(item) for item in blockers)),
        "commands": [
            "catalyst-kv-cache demo --json",
            "catalyst-kv-cache doctor",
            "catalyst-kv-cache serve --dry-run --json",
            "CATALYST_USE_WRANGLER_OAUTH=1 python -m bench.next_evidence --output site/next_evidence_results.json --chart-dir site/charts",
            "CATALYST_USE_WRANGLER_OAUTH=1 python -m bench.official_longbench_ruler --output site/official_longbench_ruler_results.json --chart-dir site/charts --run-dir /Users/ghostmesh/benchmark-runs/catalyst-rain-official --max-input-tokens 12000 --ruler-prompt-mode answer_only",
        ],
    }


def _default_evidence_root() -> Path:
    return Path(__file__).resolve().parents[2] / "evidence" / "hkvc-first-evidence-package"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
