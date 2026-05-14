from __future__ import annotations

from importlib import import_module
from typing import Any

from catalyst_kv_cache.core import CatalystKVCache, CatalystKVConfig


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
