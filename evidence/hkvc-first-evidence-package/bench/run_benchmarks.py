from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.sdk_delegate import call_catalyst_brain


def run_dropin_cache_probe() -> dict[str, Any]:
    return call_catalyst_brain(("run_dropin_cache_probe", "run_hkvc_dropin_cache_probe"))


def run_benchmarks(*, iterations: int, output: str | Path) -> dict[str, Any]:
    # catalyst_brain owns the private HKVC runtime and benchmark implementation.
    return call_catalyst_brain(
        ("run_hkvc_benchmarks", "run_catalyst_hkvc_benchmarks"),
        iterations=iterations,
        output=output,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Catalyst HKVC public evidence via catalyst-brain.")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", default="site/results.json")
    args = parser.parse_args()
    result = run_benchmarks(iterations=args.iterations, output=args.output)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
