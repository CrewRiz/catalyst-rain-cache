from __future__ import annotations

import pytest


def test_passthrough_update_preserves_values_and_records_state():
    from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig

    cache = CatalystKVCache(CatalystKVConfig(mode="passthrough", dim=512))
    keys = [[float(i), float(i + 1)] for i in range(64)]
    values = [[float(i * 2), float(i * 2 + 1)] for i in range(64)]

    out_keys, out_values = cache.update(keys, values, layer_idx=0, cache_kwargs={"token_count": 64})

    assert out_keys is keys
    assert out_values is values
    assert cache.get_seq_length(0) == 64
    report = cache.compression_report()
    assert report["records"] == 1
    assert report["original_bytes_estimate"] > 0
    assert cache.to_rain_header(agent_id="test")


def test_refs_mode_returns_compact_record_and_materializes_metadata():
    from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig

    cache = CatalystKVCache(CatalystKVConfig(mode="refs", dim=512))
    record = cache.update([1.0, -1.0], [2.0, -2.0], layer_idx=2)

    assert isinstance(record, dict)
    assert record["kv_ref"].startswith("catalyst-kv://layer/2")
    materialized = cache.materialize(record["kv_ref"])
    assert materialized is not None
    assert materialized["layer_idx"] == 2


def test_lookup_exact_fingerprint():
    from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig

    cache = CatalystKVCache(CatalystKVConfig(mode="refs", dim=512))
    key = [1.0, 2.0, 3.0]
    cache.update(key, [4.0, 5.0, 6.0], layer_idx=0)

    found = cache.lookup(layer_idx=0, query=key)

    assert found is not None
    assert found["layer_idx"] == 0


def test_commercial_purpose_requires_license():
    from catalyst_kv_cache import CatalystKVConfig, CatalystKVLicenseError

    with pytest.raises(CatalystKVLicenseError):
        CatalystKVConfig(purpose="production enterprise pilot")
