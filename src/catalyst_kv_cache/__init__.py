"""Catalyst KV-cache adapter for research and evaluation."""

from catalyst_kv_cache.core import (
    CacheRecord,
    CatalystKVCache,
    CatalystKVConfig,
    CatalystKVError,
)
from catalyst_kv_cache.license import (
    COMMERCIAL_CONTACT,
    CatalystKVLicenseError,
    assert_research_use,
)
from catalyst_kv_cache.sdk_bridge import (
    create_transformers_cache,
    load_catalyst_brain,
    onboarding_payload,
    sdk_status,
)
from catalyst_kv_cache.serve import CatalystServeConfig

__version__ = "0.1.0"

__all__ = [
    "COMMERCIAL_CONTACT",
    "CacheRecord",
    "CatalystKVCache",
    "CatalystKVConfig",
    "CatalystKVError",
    "CatalystServeConfig",
    "CatalystKVLicenseError",
    "assert_research_use",
    "create_transformers_cache",
    "load_catalyst_brain",
    "onboarding_payload",
    "sdk_status",
    "__version__",
]
