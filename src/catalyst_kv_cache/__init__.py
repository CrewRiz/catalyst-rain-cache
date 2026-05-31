"""Catalyst KV-cache adapter for research and evaluation."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_LAZY_EXPORTS = {
    "CacheRecord": ("catalyst_kv_cache.core", "CacheRecord"),
    "CatalystKVCache": ("catalyst_kv_cache.core", "CatalystKVCache"),
    "CatalystKVConfig": ("catalyst_kv_cache.core", "CatalystKVConfig"),
    "CatalystKVError": ("catalyst_kv_cache.core", "CatalystKVError"),
    "COMMERCIAL_CONTACT": ("catalyst_kv_cache.license", "COMMERCIAL_CONTACT"),
    "CatalystKVLicenseError": ("catalyst_kv_cache.license", "CatalystKVLicenseError"),
    "assert_research_use": ("catalyst_kv_cache.license", "assert_research_use"),
    "create_transformers_cache": ("catalyst_kv_cache.sdk_bridge", "create_transformers_cache"),
    "load_catalyst_brain": ("catalyst_kv_cache.sdk_bridge", "load_catalyst_brain"),
    "onboarding_payload": ("catalyst_kv_cache.sdk_bridge", "onboarding_payload"),
    "sdk_status": ("catalyst_kv_cache.sdk_bridge", "sdk_status"),
    "CatalystServeConfig": ("catalyst_kv_cache.serve", "CatalystServeConfig"),
}

__all__ = [*_LAZY_EXPORTS, "__version__"]


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module 'catalyst_kv_cache' has no attribute {name!r}") from exc
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
