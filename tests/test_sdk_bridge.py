from __future__ import annotations


def test_create_transformers_cache_falls_back_to_public_adapter():
    from catalyst_kv_cache import CatalystKVCache, create_transformers_cache

    cache = create_transformers_cache(mode="passthrough", prefer_sdk=False, dim=256)

    assert isinstance(cache, CatalystKVCache)
    assert cache.config.mode == "passthrough"


def test_sdk_status_reports_feature_names():
    from catalyst_kv_cache import sdk_status

    status = sdk_status()

    assert status["adapter_package"] == "catalyst-kv-cache"
    assert "CatalystDynamicKVCache" in status["sdk_features"]
    assert status["public_boundary"]["ships_private_algorithms"] is False


def test_create_transformers_cache_prefers_sdk_when_available(monkeypatch):
    from catalyst_kv_cache import sdk_bridge

    class FakeDynamicCache:
        def __init__(self, *, dim):
            self.dim = dim

    class FakeSDK:
        CatalystDynamicKVCache = FakeDynamicCache

    monkeypatch.setattr(sdk_bridge, "load_catalyst_brain", lambda: FakeSDK)

    cache = sdk_bridge.create_transformers_cache(mode="passthrough", dim=777)

    assert isinstance(cache, FakeDynamicCache)
    assert cache.dim == 777


def test_create_transformers_cache_falls_back_when_sdk_constructor_is_unavailable(monkeypatch):
    from catalyst_kv_cache import CatalystKVCache, sdk_bridge

    class BrokenDynamicCache:
        def __init__(self, *, dim):
            raise RuntimeError("transformers unavailable")

    class FakeSDK:
        CatalystDynamicKVCache = BrokenDynamicCache

    monkeypatch.setattr(sdk_bridge, "load_catalyst_brain", lambda: FakeSDK)

    cache = sdk_bridge.create_transformers_cache(mode="passthrough", dim=128)

    assert isinstance(cache, CatalystKVCache)
