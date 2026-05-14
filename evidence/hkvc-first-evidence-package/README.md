# Catalyst HKVC

Catalyst HKVC is a benchmarkable prototype for an edge-native long-context cache. The primary claim is fixed-state Catalyst memory plus bounded active-window execution; chunked compact archives are an optional saturation-safe storage strategy, not the hot decode path.

The public claim is intentionally bounded:

- Exact mode is a standard KV-compatible execution lane for the active window.
- Fixed Catalyst state is the O(1) memory carrier for far-context recall and handoff.
- Compact semantic mode is a lossy semantic deep-context lane, not an exact replacement for every historical KV tensor.
- Lane crossing is explicit and delegated through the Catalyst Brain SDK.
- Chunked accumulators are optional archive chunks at the `D / 9` saturation boundary. With `D=4096`, the strict safe chunk size is 455 tokens.
- RAIN (`.rain`) payloads are state-in-payload transport. Production payloads should be encrypted and authenticated.

## Current Evidence

The checked-in benchmark artifacts live in `site/`.

| Artifact | Purpose |
| --- | --- |
| `site/breakthrough_assessment.json` | Scope-aware verdict: breakthrough for fixed-state long-context recall, with exact-state lossless replacement now tracked separately from compact semantic memory. |
| `site/results.json` | Exact-lane zero-accumulator decode metrics, compact-lane saturation metrics, and Hugging Face cache shape contract. |
| `site/fixed_state_quality_results.json` | Fixed-state evidence imported from public-wheel benchmark outputs: TinyLlama effective 2M perplexity, 1M secret retrieval, O(1) exact-query latency span, and modern baseline memory comparison. |
| `site/ruler_needle_results.json` | Synthetic RULER/Needle-style suite: single needle, multi-needle, duplicate/latest-position, and wrong-position rejection against sliding-window and linear-scan baselines. |
| `site/manifold_attention_results.json` | SDK-owned attention evidence tier: fixed-state deep-context quality versus the earlier compact proxy baseline. |
| `site/lossless_equivalence_results.json` | SDK-owned exact-state/tiled-attention evidence: agreement with PyTorch scaled-dot-product attention while the local tensor window remains bounded. |
| `site/lossless_scale_results.json` | Scaled lossless evidence: multiple tensor shapes, additive masks, and a three-layer decode stack against PyTorch scaled-dot-product attention. |
| `site/tier2_evidence_results.json` | Tier-two evidence: broader fixed-corpus model-level target perplexity, LongBench/RULER-style tasks beyond needle retrieval, generation rubric scoring, and baseline efficiency rows. |
| `site/next_evidence_results.json` | Next-tier readiness artifact: official LongBench/RULER source reachability, public adapter latency, baseline import status, and Cloudflare Workers AI stronger-model probe status. |
| `site/official_longbench_ruler_results.json` | Official-source subset run: LongBench v2 data/prompt/scoring and RULER generated synthetic tasks with Cloudflare Workers AI inference. |
| `site/rain_transport_probe_results.json` | Packed-token RAIN (`.rain`) Worker transport-size probe over official LongBench v2 subset prompts. |
| `site/private_context_results.json` | Private AI long-context memory, transfer, chunk-count, and saturation estimates for 128K, 512K, and 1M tokens. |
| `site/charts/fixed_state_memory.svg` | Standard KV memory growth versus fixed Catalyst state. |
| `site/charts/quality_perplexity.svg` | TinyLlama effective 2M target perplexity with and without Catalyst recall. |
| `site/charts/retrieval_quality.svg` | 1M-token secret retrieval accuracy plus exact-key latency-span guard. |
| `site/charts/baseline_comparison.svg` | FP16, TurboQuant, KIVI, PyramidKV, and Catalyst fixed-state memory comparison. |
| `site/charts/ruler_accuracy.svg` | RULER/Needle retrieval and rejection accuracy. |
| `site/charts/ruler_latency.svg` | Catalyst fixed-state query batches compared with linear scan batches. |
| `site/charts/ruler_memory.svg` | 1M logical-token memory shape across standard KV, sliding window, linear scan, and Catalyst state. |
| `site/charts/ruler_baselines.svg` | Baseline outcome summary. |
| `site/charts/manifold_accuracy.svg` | Catalyst SDK attention top-1 retrieval versus compact proxy. |
| `site/charts/manifold_quality_gap.svg` | Cosine similarity to full softmax attention output. |
| `site/charts/manifold_memory.svg` | Standard KV bytes versus fixed SDK-owned state. |
| `site/charts/manifold_architecture.svg` | Exact-window, SDK attention state, and bounded router architecture. |
| `site/charts/lossless_equivalence.svg` | Exact-state lossless attention agreement with PyTorch SDPA. |
| `site/charts/lossless_memory.svg` | Standard KV bytes, bounded local window bytes, and exact payload bytes. |
| `site/charts/lossless_scale_accuracy.svg` | Scaled exact-state lossless agreement across decode, prefill, mask, and layer-stack profiles. |
| `site/charts/lossless_scale_memory.svg` | Memory placement for the scaled lossless profiles. |
| `site/charts/lossless_scale_latency.svg` | Prototype timing across scaled lossless profiles. |
| `site/charts/tier2_perplexity.svg` | Model-level target perplexity on fixed corpora with retrieved context materialized into the prompt. |
| `site/charts/tier2_longbench_ruler.svg` | QA, code, timeline, variable-tracking, and multi-hop fixed-corpus task accuracy. |
| `site/charts/tier2_generation_quality.svg` | Fixed-prompt generation rubric scores. |
| `site/charts/tier2_baseline_memory.svg` | Memory comparison against KIVI, TurboQuant, PyramidKV/H2O, vLLM/PagedAttention, sparse retrieval, and Catalyst. |
| `site/charts/tier2_latency_throughput.svg` | Measured Catalyst prototype throughput row versus modeled baseline rows. |
| `site/charts/next_official_readiness.svg` | Official benchmark source reachability and score-readiness status. |
| `site/charts/next_adapter_latency.svg` | Measured public adapter update latency for passthrough and reference modes. |
| `site/charts/next_baseline_imports.svg` | Import readiness for modern baseline implementations. |
| `site/charts/next_cloudflare_ai.svg` | Cloudflare Workers AI account/token/measurement status. |
| `site/charts/official_longbench_accuracy.svg` | Official LongBench v2 subset accuracy. |
| `site/charts/official_ruler_accuracy.svg` | Official RULER generated-subset task scores. |
| `site/charts/rain_worker_transport.svg` | Packed-token/RAIN Worker transport boundary for larger-context benchmark runs. |
| `site/charts/rain_transport_payload.svg` | Raw chat JSON bytes versus packed-token Worker JSON bytes over the official LongBench subset. |
| `site/charts/breakthrough_verdict.svg` | Evidence-score verdict across claim categories. |
| `site/charts/claim_scope_matrix.svg` | Safe public-claim matrix. |
| `site/charts/memory_scaling.svg` | Standard KV memory vs chunked HKVC deep-context memory. |
| `site/charts/transfer_time.svg` | Estimated payload transfer time at 32 GB/s. |
| `site/charts/saturation_boundary.svg` | Single-accumulator saturation vs chunked accumulator noise floor. |
| `site/charts/edge_native_architecture.svg` | Exact-window, promotion, chunked-HKVC, and RAIN handoff diagram. |
| `site/index.html` | Static evidence page that loads the JSON and renders the charts. |

Under the fixed-state evidence profile, an 8B/GQA-style 1M-token KV cache is estimated at 131,072,000,000 bytes, while the Catalyst far-context state is 16,384 bytes and the bounded 1,024-token hot cache is 134,234,112 bytes. Existing public-wheel results report 100% 1M-token 256-bit secret retrieval over 32 trials and TinyLlama effective 2M target perplexity improving from 27.795727 to 1.086565 when Catalyst recall is materialized into the active prompt.

The RULER/Needle-style suite reports 100% Catalyst retrieval accuracy across single-needle, multi-needle, and duplicate/latest-position tasks at a 1M-token logical context. The sliding-window baseline reports 0% far-context retrieval because those needles sit outside the recent 1,024-token window, and a recent-window positive control reports 100% sliding-window retrieval. Catalyst also reports 0% false positives on wrong-position rejection. Linear scan is included as a retrieval baseline, but is labeled non-O(1) because its memory and work grow with retained records.

The attention tier is the first SDK-owned gap-closing experiment toward full KV-cache replacement. The public repo records the benchmark inputs, outputs, charts, and claim boundaries, while the retrieval operator itself lives behind the `catalyst-brain` SDK boundary. In the checked-in run, the Catalyst SDK path reaches 100% top-1 retrieval over 16 far-context trials, the compact proxy reaches 25%, and the SDK path has 0.833 mean cosine to the full attention reference output versus 0.440 for the compact proxy. This is still a synthetic operator benchmark, not a model-level perplexity proof.

The lossless tier now covers decode overflow, causal prefill, and additive-mask attention cases. The checked-in run reports 24 scenarios with 100% allclose agreement to PyTorch scaled-dot-product attention, max absolute error under `1e-6`, and a bounded local tensor window that remains smaller than the logical exact state. This evidence supports the exact-state/tiled-attention replacement path; it does not make the compact semantic lane a lossless finite-dimensional encoding of arbitrary KV tensors.

The scaled lossless tier broadens the same claim across four profiles: decode overflow, causal prefill, additive-mask attention, and a three-layer decode stack. The public artifact records the profile metadata, agreement numbers, local-window memory, exact payload bytes, and Python timing. It is still operator-level evidence, not a full model-generation benchmark.

The tier-two evidence package starts closing those gaps. It adds six fixed-corpus tasks across contract QA, code configuration, operations, timeline lookup, variable tracking, and multi-hop policy. The checked-in run now uses `distilgpt2` for target-answer perplexity, reports 83.33% Catalyst retrieval accuracy, 0% sliding-window fixed-corpus accuracy, 86.67% mean generation-rubric score, and a 4.285x target-PPL improvement when retrieved context is materialized. This is stronger than the earlier tiny-model smoke row, but it is still a fixed-corpus harness rather than an official LongBench/RULER submission.

The baseline efficiency rows now include KIVI 2-bit, TurboQuant 3.5-bit, PyramidKV 12%, H2O 20%, vLLM/PagedAttention, and sparse/retrieval attention. Those rows are modeled unless explicitly marked measured. The Catalyst row is measured through the pure-Python prototype and currently shows memory advantage but poor Python latency; production kernel throughput remains future evidence.

The next-tier readiness artifact tracks the remaining launch-grade evidence work without overstating it. In the checked-in run, LongBench and RULER official sources are reachable, public adapter update latency is measured over 512 updates, no modern baseline package is installed locally yet, and Cloudflare Workers AI reports account reachability but skips the REST quality probe because `CLOUDFLARE_API_TOKEN` is not present in the benchmark shell. That Cloudflare row is readiness status, not model-quality evidence.

The official-source subset artifact now runs real LongBench/RULER evidence instead of only readiness checks. The checked-in run uses LongBench v2 official data and prompt format over a 36-example stratified subset, with a 12K-token prompt cap because Workers AI direct REST returned payload-size errors at 32K and 64K in this environment. It reports 36.11% LongBench v2 subset accuracy. The RULER run uses NVIDIA RULER synthetic data generation for six no-external-data tasks at 4K and 8K, two samples per task/length, and reports a 97.22% mean score over 24 predictions. The earlier 0% variable-tracking row was a prompt-format/answer-budget failure: the runner now supports an explicitly labeled `answer_only` mode that appends the official RULER `answer_prefix`, moving variable tracking to 100% at both 4K and 8K. This is an official-source subset, not a full official submission and not a Catalyst adapter quality number until the same runner is connected to a live Catalyst/catalyst-brain model path.

The official runner also includes a Catalyst RAIN (`.rain`) Worker execution path with a packed-token context transport. In that mode, this public adapter sends tokenizer metadata, a prompt hash, and a reversible compressed token payload to a Worker; binding, lossless state handling, retrieval, rehydration, and active-context materialization remain in `catalyst-brain`. This can avoid raw-prompt payload failures, but it does not bypass the target model's real attention limit unless the Worker materializes a valid shorter active context.

The RAIN transport probe measures that envelope on the same LongBench v2 subset with local truncation disabled. The checked-in run reports mean packed Worker JSON at 35.59% of raw Cloudflare chat JSON, with max raw JSON at 4,681,132 bytes and max packed Worker JSON at 1,271,763 bytes. That supports the transport-bypass path; it is not model-quality evidence.

The chunked archive benchmark remains useful for a saturation-safe semantic archive, but it is not the headline O(1) decode claim.

## Breakthrough Verdict

Yes, with scope. The evidence supports calling Catalyst HKVC a breakthrough in fixed-state private long-context memory: far-context recall remains in a 16,384-byte Catalyst state while the hot cache stays bounded, and the published quality/retrieval artifacts show useful behavior rather than memory math alone.

Do not call the compact semantic lane a universal lossless KV-cache replacement. The new lossless tier supports a different claim: universal exactness requires exact state transport or hydration plus an attention replacement that consumes that exact state in bounded tiles.

## Run Tests

From the repository root:

```bash
python3 -m pytest catalyst_brain/tests/test_hkvc_spec_runtime.py -q
python3 -m pytest catalyst_brain/tests/test_dropin_adapters.py -q
python3 -m py_compile catalyst-hkvc/sdk/*.py catalyst-hkvc/bench/*.py catalyst_brain/tests/test_hkvc_spec_runtime.py
```

## Regenerate Benchmarks

From `catalyst-hkvc/`:

```bash
catalyst-kv-cache doctor
python3 -m bench.run_benchmarks --iterations 1000 --output site/results.json
python3 -m bench.fixed_state_quality --source-results /Users/ghostmesh/catalyst-brain-benchmarks/results --output site/fixed_state_quality_results.json --chart-dir site/charts
python3 -m bench.ruler_needle_benchmark --output site/ruler_needle_results.json --chart-dir site/charts --trials 32 --distractors 2048
python3 -m bench.manifold_attention_benchmark --output site/manifold_attention_results.json --chart-dir site/charts --trials 16 --tokens 256 --dim 32 --feature-dim 512
python3 -m bench.lossless_equivalence --output site/lossless_equivalence_results.json --chart-dir site/charts --trials 8 --tokens 32 --heads 2 --head-dim 16 --max-tokens 4 --tile-tokens 8
python3 -m bench.lossless_scale --output site/lossless_scale_results.json --chart-dir site/charts --trials 2 --max-tokens 8 --tile-tokens 16
python3 -m bench.tier2_evidence --output site/tier2_evidence_results.json --chart-dir site/charts --model-name distilgpt2 --tokens 128000
CLOUDFLARE_API_TOKEN="${CLOUDFLARE_API_TOKEN:?set a Cloudflare token}" python3 -m bench.next_evidence --output site/next_evidence_results.json --chart-dir site/charts --updates 512
CLOUDFLARE_API_TOKEN="${CLOUDFLARE_API_TOKEN:?set a Cloudflare token}" python3 -m bench.official_longbench_ruler --output site/official_longbench_ruler_results.json --chart-dir site/charts --run-dir /Users/ghostmesh/benchmark-runs/catalyst-kv-cache-official --longbench-limit 36 --max-input-tokens 12000 --ruler-tasks niah_single_1,niah_multikey_2,niah_multikey_3,vt,cwe,fwe --ruler-lengths 4096,8192 --ruler-samples 2 --ruler-prompt-mode answer_only
python3 -m bench.rain_transport_probe --output site/rain_transport_probe_results.json --chart-dir site/charts --longbench-limit 36
python3 -m bench.private_context_benchmark --output site/private_context_results.json --chart-dir site/charts
python3 -m bench.breakthrough_assessment --site-dir site --output site/breakthrough_assessment.json --chart-dir site/charts
```

For the Catalyst RAIN (`.rain`) Worker transport path, point the same runner at a Worker
that consumes `catalyst-brain` privately:

```bash
CATALYST_RAIN_WORKER_URL="https://<worker>/bench" \
CATALYST_RAIN_WORKER_API_KEY="${CATALYST_RAIN_WORKER_API_KEY:?set worker key}" \
python3 -m bench.official_longbench_ruler \
  --execution-path catalyst_rain_worker \
  --rain-context-transport packed_tokens \
  --output site/official_longbench_ruler_results.json \
  --chart-dir site/charts \
  --run-dir /Users/ghostmesh/benchmark-runs/catalyst-kv-cache-official \
  --longbench-limit 36 \
  --max-input-tokens 0 \
  --ruler-prompt-mode answer_only
```

The official LongBench/RULER subset runner expects local official clones at `/Users/ghostmesh/benchmark-runs/industry/LongBench` and `/Users/ghostmesh/benchmark-runs/industry/RULER`, plus RULER Git LFS assets pulled. It intentionally keeps benchmark dependencies out of the core adapter install; the evidence environment used `datasets`, `tiktoken`, `nltk`, `pandas`, `wonderwords`, and `scipy`.

The benchmark entrypoints above delegate private runtime/evidence work to the
installed `catalyst-brain` SDK. If `doctor` reports missing evidence exports,
install an SDK build that includes the HKVC evidence runner surface before
regenerating charts.

Serve the static page locally so the browser can fetch the JSON artifacts:

```bash
cd site
python3 -m http.server 8027
```

Then open `http://127.0.0.1:8027/`.

## SDK Boundary

This public package is an adapter and evidence repo. It does not ship Catalyst
Brain algorithm implementations. Benchmark entrypoints delegate private runtime,
attention, and cache generation work to the installed `catalyst-brain` SDK, then
publish only reproducible artifacts, charts, and claim metadata.

Public code may expose thin wrappers such as `bench.tier2_evidence`,
`bench.lossless_equivalence`, `bench.lossless_scale`, and
`bench.manifold_attention_benchmark`; the private implementation belongs in
the SDK.

## Evidence Still Needed

This repo now publishes runtime invariants, fixed-state memory math, TinyLlama effective-context perplexity, 1M retrieval, saturation boundaries, transfer estimates, drop-in tensor-shape bridge evidence, broader fixed-corpus retrieval, model-level PPL harness output, generation rubric scoring, and expanded baseline efficiency comparisons. Remaining evidence gaps:

- Rerun `bench.tier2_evidence` with a live Catalyst API path or a larger instruction model; the checked-in `distilgpt2` PPL row is measured locally but still small-model evidence.
- Provide `CLOUDFLARE_API_TOKEN` to rerun the Cloudflare Workers AI quality probe, then expand it to fixed-prompt generation scoring and official benchmark runners.
- Connect the Catalyst RAIN (`.rain`) Worker path to a live `catalyst-brain` backend, then rerun LongBench v2 without the 12K raw-prompt cap.
- Expand the official-source LongBench/RULER subset run to all 503 LongBench v2 examples, the full RULER task matrix, and larger context caps through the RAIN Worker or another endpoint that accepts state transport.
- Replace modeled KIVI, TurboQuant, PyramidKV/H2O, vLLM/PagedAttention, and sparse/retrieval rows with imported implementation runs on the same hardware.
- Add production kernel latency/throughput for the SDK-owned attention path; the current Catalyst latency row is pure Python.

Those stronger quality numbers should be added only after the benchmark harness connects to the target model/API and records reproducible run metadata.
