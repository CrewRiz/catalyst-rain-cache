from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from bench.sdk_delegate import call_catalyst_brain


DEFAULT_TOKENS = 128_000


def run_tier2_evidence(
    *,
    output: str | Path,
    chart_dir: str | Path,
    model_name: str = "distilgpt2",
    run_model: bool = True,
    dim: int = 128,
    feature_dim: int = 256,
    tokens: int = DEFAULT_TOKENS,
) -> dict[str, Any]:
    # catalyst_brain owns private retrieval, cache, and evidence-generation code.
    return call_catalyst_brain(
        ("run_hkvc_tier2_evidence", "run_tier2_evidence"),
        output=output,
        chart_dir=chart_dir,
        model_name=model_name,
        run_model=run_model,
        dim=dim,
        feature_dim=feature_dim,
        tokens=tokens,
    )


def score_generation_rubric(
    *,
    answer: str,
    required_facts: Sequence[str],
    forbidden_facts: Sequence[str],
) -> dict[str, Any]:
    answer_lower = answer.lower()
    missing = [fact for fact in required_facts if fact.lower() not in answer_lower]
    forbidden = [fact for fact in forbidden_facts if fact.lower() in answer_lower]
    required_score = (len(required_facts) - len(missing)) / max(1, len(required_facts))
    penalty = len(forbidden) / max(1, len(forbidden_facts))
    return {
        "score": max(0.0, round(0.8 * required_score - 0.2 * penalty + 0.2, 4)),
        "missing_required": missing,
        "forbidden_present": forbidden,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run tier-two HKVC evidence via catalyst-brain.")
    parser.add_argument("--output", default="site/tier2_evidence_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--model-name", default="distilgpt2")
    parser.add_argument("--tokens", type=int, default=DEFAULT_TOKENS)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=256)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    payload = run_tier2_evidence(
        output=args.output,
        chart_dir=args.chart_dir,
        model_name=args.model_name,
        run_model=not args.skip_model,
        dim=args.dim,
        feature_dim=args.feature_dim,
        tokens=args.tokens,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
