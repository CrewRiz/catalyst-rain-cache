# Transformers Integration Notes

`CatalystKVCache.update(key_states, value_states, layer_idx, cache_kwargs=None)`
matches the common Hugging Face cache update shape closely enough for research
adapters.

Recommended integration path:

1. Start with `mode="passthrough"` so model output remains unchanged.
2. Record Catalyst compression reports and Rain headers next to normal runs.
3. Compare latency, memory model, and quality metrics.
4. Move to `mode="refs"` only inside a serving stack that knows how to consume
   compact Catalyst references.

Production or enterprise integrations require a written license or pilot
agreement. Contact `hello@strategic-innovations.ai`.
