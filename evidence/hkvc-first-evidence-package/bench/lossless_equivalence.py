from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.sdk_delegate import call_catalyst_brain


def run_lossless_equivalence(
    *,
    output: str | Path,
    chart_dir: str | Path,
    trials: int = 8,
    tokens: int = 32,
    heads: int = 2,
    head_dim: int = 16,
    max_tokens: int = 4,
    tile_tokens: int = 8,
) -> dict[str, Any]:
    # catalyst_brain owns the exact-state and attention-equivalence implementation.
    return call_catalyst_brain(
        ("run_hkvc_lossless_equivalence_benchmark", "run_lossless_equivalence_benchmark"),
        output=output,
        chart_dir=chart_dir,
        trials=trials,
        tokens=tokens,
        heads=heads,
        head_dim=head_dim,
        max_tokens=max_tokens,
        tile_tokens=tile_tokens,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run lossless KV equivalence evidence via catalyst-brain.")
    parser.add_argument("--output", default="site/lossless_equivalence_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--trials", type=int, default=8)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=4)
    parser.add_argument("--tile-tokens", type=int, default=8)
    args = parser.parse_args()
    payload = run_lossless_equivalence(
        output=args.output,
        chart_dir=args.chart_dir,
        trials=args.trials,
        tokens=args.tokens,
        heads=args.heads,
        head_dim=args.head_dim,
        max_tokens=args.max_tokens,
        tile_tokens=args.tile_tokens,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
