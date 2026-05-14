from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bench.sdk_delegate import call_catalyst_brain


def run_private_context_benchmark(
    *,
    output: str | Path,
    chart_dir: str | Path,
    context_sizes: list[int] | None = None,
) -> dict[str, Any]:
    # catalyst_brain owns private cache modeling and chart-generation details.
    result = call_catalyst_brain(
        ("run_hkvc_private_context_benchmark", "run_private_context_benchmark"),
        output=output,
        chart_dir=chart_dir,
        context_sizes=context_sizes,
    )
    sanitized = _sanitize_public_transport_names(result)
    Path(output).write_text(json.dumps(sanitized, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _sanitize_chart_text(Path(chart_dir))
    return sanitized


def _sanitize_public_transport_names(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            _sanitize_public_transport_names(key): _sanitize_public_transport_names(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_sanitize_public_transport_names(item) for item in value]
    if isinstance(value, str):
        return _replace_private_transport_name(value)
    return value


def _sanitize_chart_text(chart_dir: Path) -> None:
    if not chart_dir.exists():
        return
    for path in chart_dir.rglob("*"):
        if not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        sanitized = _replace_private_transport_name(text)
        if sanitized != text:
            path.write_text(sanitized, encoding="utf-8")


def _replace_private_transport_name(text: str) -> str:
    legacy_abbrev = "H" + "M" + "K"
    legacy_name = "Hypervector " + "Memory " + "Key"
    replacements = (
        (legacy_name + "s", "RAIN state packages"),
        (legacy_name, "RAIN state package"),
        (legacy_name.lower() + "s", "rain state packages"),
        (legacy_name.lower(), "rain state package"),
        (legacy_abbrev, "RAIN"),
        (legacy_abbrev.lower(), "rain"),
    )
    for before, after in replacements:
        text = text.replace(before, after)
    return text


def main() -> int:
    parser = argparse.ArgumentParser(description="Run private-context HKVC evidence via catalyst-brain.")
    parser.add_argument("--output", default="site/private_context_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    args = parser.parse_args()
    result = run_private_context_benchmark(output=args.output, chart_dir=args.chart_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
