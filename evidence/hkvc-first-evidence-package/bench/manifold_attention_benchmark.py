from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.sdk_delegate import call_catalyst_brain


def run_manifold_attention_benchmark(
    *,
    output: str | Path,
    chart_dir: str | Path,
    trials: int = 32,
    tokens: int = 512,
    dim: int = 64,
    feature_dim: int = 512,
) -> dict[str, Any]:
    # catalyst_brain owns the private attention operator and chart generator.
    return call_catalyst_brain(
        ("run_hkvc_attention_benchmark", "run_catalyst_manifold_attention_benchmark"),
        output=output,
        chart_dir=chart_dir,
        trials=trials,
        tokens=tokens,
        dim=dim,
        feature_dim=feature_dim,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Catalyst attention evidence via catalyst-brain.")
    parser.add_argument("--output", default="site/manifold_attention_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--feature-dim", type=int, default=512)
    args = parser.parse_args()
    payload = run_manifold_attention_benchmark(
        output=args.output,
        chart_dir=args.chart_dir,
        trials=args.trials,
        tokens=args.tokens,
        dim=args.dim,
        feature_dim=args.feature_dim,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
