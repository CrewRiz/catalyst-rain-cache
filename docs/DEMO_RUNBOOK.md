# Demo Runbook

Use this when walking a customer, investor, or internal reviewer through the
public RAIN cache adapter evidence package.

## One-Command Summary

```bash
catalyst-kv-cache demo
catalyst-kv-cache demo --json
```

The demo command reads the checked-in evidence artifacts and reports:

- scoped breakthrough verdict
- official LongBench v2 subset result
- official RULER generated subset result
- live Cloudflare Workers AI probe status
- RAIN packed-token transport size evidence
- remaining blockers that are not yet claimed

## Live Cloudflare Probe

If `wrangler whoami` shows an OAuth session with `ai` permissions, the evidence
runner can use that local session without printing or committing credentials:

```bash
CATALYST_USE_WRANGLER_OAUTH=1 \
python -m bench.next_evidence \
  --output site/next_evidence_results.json \
  --chart-dir site/charts
```

An explicit `CLOUDFLARE_API_TOKEN` still works. The public artifact records only
the auth source and probe result, not the token.

## Official Subset Refresh

The current public package publishes a subset artifact, not a full leaderboard
submission:

```bash
CATALYST_USE_WRANGLER_OAUTH=1 \
python -m bench.official_longbench_ruler \
  --output site/official_longbench_ruler_results.json \
  --chart-dir site/charts \
  --run-dir /Users/ghostmesh/benchmark-runs/catalyst-rain-official \
  --longbench-limit 36 \
  --max-input-tokens 12000 \
  --ruler-samples 2 \
  --ruler-prompt-mode answer_only
```

Claim boundary for the talk track: this demonstrates the official runner/data
path and live stronger-model execution. It is not a Catalyst adapter quality
claim until inference runs through the private `catalyst-brain` RAIN serving
path.

## Demo Talk Track

1. Start with `catalyst-kv-cache demo`.
2. Open `evidence/hkvc-first-evidence-package/site/index.html` for charts.
3. Show `catalyst-kv-cache doctor` to prove the public adapter delegates private
   algorithms to `catalyst-brain`.
4. Show `catalyst-kv-cache serve --dry-run --json` for the OpenAI-compatible
   serving shape.
5. Be explicit: fixed-state private long-context memory and stateless handoff
   are the current evidence-backed wedge; universal replacement is tracked via
   the exact-state SDK path, not by claiming fixed-size semantic compression is
   magically lossless.
