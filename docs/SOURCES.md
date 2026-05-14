# KV-Cache Comparison Sources

The README chart is a simple memory-scaling model. It compares linear KV-cache
compression systems against Catalyst Brain's fixed public SDK state.

Sources used:

- TurboQuant: Google Research / arXiv, "Online Vector Quantization for
  Efficient High-Throughput LLM Inference" (`arXiv:2504.19874`,
  https://arxiv.org/abs/2504.19874). The chart uses the reported
  3.5 bits/channel quality-neutral point as the modeled memory ratio versus
  FP16.
- KIVI: "A Tuning-Free Asymmetric 2bit Quantization for KV Cache"
  (`arXiv:2402.02750`, https://arxiv.org/abs/2402.02750). The chart models raw
  2-bit KV storage versus FP16.
- PyramidKV: "Dynamic KV Cache Compression based on Pyramidal Information
  Funneling" (`arXiv:2406.02069`, https://arxiv.org/abs/2406.02069). The chart
  uses the paper's 12% KV retention setting.
- H2O: "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large
  Language Models" (`arXiv:2306.14048`, https://arxiv.org/abs/2306.14048).
  Tier-two evidence models a 20% retained-cache row to represent
  heavy-hitter/recent-token retention.
- vLLM/PagedAttention: "Efficient Memory Management for Large Language Model
  Serving with PagedAttention" (`arXiv:2309.06180`,
  https://arxiv.org/abs/2309.06180) and the vLLM design docs
  (https://docs.vllm.ai/). Tier-two evidence models PagedAttention as an
  allocation/serving baseline that still retains FP16-style KV content.
- Sparse/retrieval attention: tier-two evidence includes a modeled top-k
  retrieval-attention row as a non-KV-cache comparison class. It is a memory and
  work model, not a specific vendor implementation.
- Catalyst Brain HKVC: fixed 4096-dim state measured through the public
  `catalyst-brain` SDK APIs in `scripts/generate_scaling_chart.py`.

This repository deliberately exposes only adapter code, public benchmark models,
and public SDK calls. The closed-source Catalyst Brain SDK remains the engine.
