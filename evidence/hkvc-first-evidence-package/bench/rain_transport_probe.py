from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from typing import Any, Iterable, Sequence

from bench.official_longbench_ruler import (
    DEFAULT_LONGBENCH_LIMIT,
    DEFAULT_LONGBENCH_REPO,
    DEFAULT_MODEL,
    _git_revision,
    _load_longbench_items,
    _longbench_prompt,
    pack_token_context,
)


def run_rain_transport_probe(
    *,
    output: str | Path,
    chart_dir: str | Path,
    longbench_repo: str | Path = DEFAULT_LONGBENCH_REPO,
    model: str = DEFAULT_MODEL,
    longbench_limit: int = DEFAULT_LONGBENCH_LIMIT,
    longbench_items: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    chart_path = Path(chart_dir)
    longbench_path = Path(longbench_repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.mkdir(parents=True, exist_ok=True)

    template = (longbench_path / "prompts" / "0shot.txt").read_text(encoding="utf-8")
    items = list(longbench_items) if longbench_items is not None else _load_longbench_items(
        longbench_repo=longbench_path,
        limit=longbench_limit,
    )
    rows = []
    for item in items:
        prompt = _longbench_prompt(template, item)
        packed = pack_token_context(prompt)
        raw_body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": 128,
        }
        worker_body = {
            "adapter": "catalyst-kv-cache",
            "execution_path": "catalyst_rain_worker",
            "context_transport": "packed_tokens",
            "private_algorithms_live_in": "catalyst-brain",
            "model": model,
            "max_tokens": 128,
            **packed,
        }
        raw_json_bytes = _json_bytes(raw_body)
        packed_json_bytes = _json_bytes(worker_body)
        rows.append(
            {
                "id": item["_id"],
                "domain": item["domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "token_count": packed["token_count"],
                "raw_prompt_bytes": len(prompt.encode("utf-8")),
                "raw_cloudflare_json_bytes": raw_json_bytes,
                "packed_worker_json_bytes": packed_json_bytes,
                "packed_vs_raw_json_ratio": round(packed_json_bytes / max(1, raw_json_bytes), 6),
                "local_truncation_disabled": True,
            }
        )

    payload = {
        "system_info": {
            "suite": "rain_transport_probe",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": model,
            "longbench_limit": longbench_limit,
            "longbench_repo": str(longbench_path),
            "longbench_revision": _git_revision(longbench_path),
            "context_transport": "packed_tokens",
            "local_truncation_disabled": True,
        },
        "claim_boundaries": {
            "transport_only_not_model_quality": True,
            "packed_token_transport_is_not_model_context_extension": True,
            "worker_must_materialize_valid_active_context": True,
            "private_algorithms_remain_in": "catalyst_brain",
        },
        "aggregate": _aggregate(rows),
        "rows": rows,
        "charts": {"rain_transport_payload": "charts/rain_transport_payload.svg"},
    }
    (chart_path / "rain_transport_payload.svg").write_text(_payload_chart(payload), encoding="utf-8")
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def _aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_count": len(rows),
        "mean_token_count": _safe_mean(row["token_count"] for row in rows),
        "mean_raw_cloudflare_json_bytes": _safe_mean(row["raw_cloudflare_json_bytes"] for row in rows),
        "mean_packed_worker_json_bytes": _safe_mean(row["packed_worker_json_bytes"] for row in rows),
        "mean_packed_vs_raw_json_ratio": _safe_mean(row["packed_vs_raw_json_ratio"] for row in rows),
        "max_raw_cloudflare_json_bytes": max((row["raw_cloudflare_json_bytes"] for row in rows), default=0),
        "max_packed_worker_json_bytes": max((row["packed_worker_json_bytes"] for row in rows), default=0),
    }


def _json_bytes(value: dict[str, Any]) -> int:
    return len(json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8"))


def _safe_mean(values: Iterable[Any]) -> float:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.fmean(numeric), 6) if numeric else 0.0


def _payload_chart(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    raw = float(aggregate["mean_raw_cloudflare_json_bytes"])
    packed = float(aggregate["mean_packed_worker_json_bytes"])
    values = [
        ("Raw JSON", raw, "#475569"),
        ("Packed Worker", packed, "#0f766e"),
    ]
    width = 760
    height = 430
    max_value = max(raw, packed, 1.0)
    body = []
    for index, (label, value, color) in enumerate(values):
        x = 180 + index * 190
        bar_height = max(3.0, 235.0 * value / max_value)
        body.append(f'<rect x="{x}" y="{330 - bar_height:.1f}" width="82" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
        body.append(f'<text x="{x + 41}" y="{max(20, 320 - bar_height):.1f}" text-anchor="middle" font-size="11" fill="#111827">{_format_bytes(value)}</text>')
        body.append(f'<text x="{x + 41}" y="364" text-anchor="middle" font-size="12" fill="#111827">{label}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        '<text x="56" y="44" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">RAIN Worker Transport Payload</text>'
        '<text x="56" y="70" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">Mean request bytes for official LongBench prompts; transport only, not model quality.</text>'
        '<line x1="70" y1="330" x2="690" y2="330" stroke="#d1d5db"/>'
        f'{"".join(body)}</svg>\n'
    )


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    current = float(value)
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f"{current:.1f} {unit}"
        current /= 1024.0
    return f"{current:.1f} GiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure packed-token RAIN Worker transport size.")
    parser.add_argument("--output", default="site/rain_transport_probe_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--longbench-repo", default=str(DEFAULT_LONGBENCH_REPO))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--longbench-limit", type=int, default=DEFAULT_LONGBENCH_LIMIT)
    args = parser.parse_args()
    payload = run_rain_transport_probe(
        output=args.output,
        chart_dir=args.chart_dir,
        longbench_repo=args.longbench_repo,
        model=args.model,
        longbench_limit=args.longbench_limit,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
