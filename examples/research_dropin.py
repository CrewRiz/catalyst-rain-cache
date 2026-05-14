from catalyst_kv_cache import CatalystKVCache, CatalystKVConfig


cache = CatalystKVCache(CatalystKVConfig(mode="passthrough", dim=1024))

key_states = [[float(i), float(i + 1)] for i in range(128)]
value_states = [[float(i * 2), float(i * 2 + 1)] for i in range(128)]

key_states, value_states = cache.update(
    key_states=key_states,
    value_states=value_states,
    layer_idx=0,
    cache_kwargs={"position": 0, "token_count": 128},
)

print(cache.compression_report())
print(cache.to_rain_header(agent_id="research-dropin")[:96] + "...")
