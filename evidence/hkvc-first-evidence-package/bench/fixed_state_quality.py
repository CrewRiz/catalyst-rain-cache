from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


DEFAULT_SOURCE_RESULTS = Path("/Users/ghostmesh/catalyst-brain-benchmarks/results")
CATALYST_STATE_BYTES = 16_384
ACTIVE_WINDOW_TOKENS = 1_024
FIXED_STATE_TOKENS = [128_000, 512_000, 1_000_000, 2_000_000]


def run_fixed_state_quality_evidence(
    *,
    source_results: str | Path = DEFAULT_SOURCE_RESULTS,
    output: str | Path,
    chart_dir: str | Path,
) -> dict[str, Any]:
    source = Path(source_results)
    tinyllama = _first_csv_row(source / "tinyllama_effective_2m_context.csv")
    secret = _first_csv_row(source / "million_token_secret_retrieval.csv")
    hkvc_scaling = _read_csv(source / "hkvc_scaling.csv")
    baselines = _read_csv(source / "kv_cache_comparison.csv")

    catalyst_state_bytes = int(tinyllama.get("catalyst_state_bytes") or secret.get("catalyst_state_bytes") or CATALYST_STATE_BYTES)
    memory_profiles = _memory_profiles(catalyst_state_bytes)
    payload = {
        "source_results": str(source),
        "claim_boundaries": {
            "fixed_state_claim": True,
            "decode_path_scans_compact_archive": False,
            "compact_archive_is_optional_not_hot_decode_state": True,
            "tinyllama_effective_context_not_raw_rope_extension": True,
            "quality_numbers_are_from_existing_public_wheel_benchmarks": True,
        },
        "memory_profiles": memory_profiles,
        "fixed_state_scaling": _fixed_state_scaling(catalyst_state_bytes, memory_profiles["llama8b_gqa"]),
        "tinyllama_effective_2m": tinyllama,
        "million_token_secret_retrieval": secret,
        "hkvc_query_scaling": _query_scaling_summary(hkvc_scaling),
        "modern_baselines": _baseline_summary(baselines),
        "charts": {
            "fixed_state_memory": "charts/fixed_state_memory.svg",
            "quality_perplexity": "charts/quality_perplexity.svg",
            "retrieval_quality": "charts/retrieval_quality.svg",
            "baseline_comparison": "charts/baseline_comparison.svg",
        },
    }
    output_path = Path(output)
    chart_path = Path(chart_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.mkdir(parents=True, exist_ok=True)
    _write_charts(payload, chart_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="") as handle:
        return [_coerce_row(row) for row in csv.DictReader(handle)]


def _first_csv_row(path: Path) -> dict[str, Any]:
    rows = _read_csv(path)
    return rows[0] if rows else {}


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    return {key: _coerce_value(value) for key, value in row.items()}


def _coerce_value(value: str) -> Any:
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        if value.strip() and value.strip() == str(int(value)):
            return int(value)
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return value


def _kv_bytes(*, tokens: int, layers: int, kv_heads: int, head_dim: int, precision_bytes: int) -> int:
    return int(tokens * layers * kv_heads * head_dim * 2 * precision_bytes)


def _memory_profiles(catalyst_state_bytes: int) -> dict[str, Any]:
    llama8b = {
        "label": "8B GQA KV profile",
        "layers": 32,
        "kv_heads": 8,
        "head_dim": 128,
        "precision_bytes": 2,
        "active_window_tokens": ACTIVE_WINDOW_TOKENS,
        "catalyst_state_bytes": catalyst_state_bytes,
    }
    full_mha = {
        "label": "MHA KV stress profile",
        "layers": 32,
        "kv_heads": 32,
        "head_dim": 128,
        "precision_bytes": 2,
        "active_window_tokens": ACTIVE_WINDOW_TOKENS,
        "catalyst_state_bytes": catalyst_state_bytes,
    }
    for profile in (llama8b, full_mha):
        standard = _kv_bytes(tokens=1_000_000, **_kv_args(profile))
        active = _kv_bytes(tokens=ACTIVE_WINDOW_TOKENS, **_kv_args(profile))
        profile["standard_kv_bytes_1m"] = standard
        profile["standard_kv_gb_1m"] = standard / 1_000_000_000.0
        profile["active_window_kv_bytes"] = active
        profile["bounded_hot_cache_bytes"] = active + catalyst_state_bytes
        profile["bounded_hot_cache_gb"] = (active + catalyst_state_bytes) / 1_000_000_000.0
        profile["state_only_reduction_vs_standard_x"] = standard / catalyst_state_bytes
        profile["bounded_hot_cache_reduction_vs_standard_x"] = standard / (active + catalyst_state_bytes)
    return {"llama8b_gqa": llama8b, "full_mha": full_mha}


def _kv_args(profile: dict[str, Any]) -> dict[str, int]:
    return {
        "layers": int(profile["layers"]),
        "kv_heads": int(profile["kv_heads"]),
        "head_dim": int(profile["head_dim"]),
        "precision_bytes": int(profile["precision_bytes"]),
    }


def _fixed_state_scaling(catalyst_state_bytes: int, profile: dict[str, Any]) -> list[dict[str, Any]]:
    active_bytes = int(profile["active_window_kv_bytes"])
    return [
        {
            "tokens": tokens,
            "standard_kv_bytes": _kv_bytes(tokens=tokens, **_kv_args(profile)),
            "catalyst_state_bytes": catalyst_state_bytes,
            "bounded_hot_cache_bytes": active_bytes + catalyst_state_bytes,
            "active_window_tokens": ACTIVE_WINDOW_TOKENS,
        }
        for tokens in FIXED_STATE_TOKENS
    ]


def _query_scaling_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    medians = [float(row["median_us"]) for row in rows if "median_us" in row]
    entries = [int(row["entries"]) for row in rows if "entries" in row]
    value_ok = all(bool(row.get("value_ok")) for row in rows)
    return {
        "rows": rows,
        "largest_entries": max(entries, default=0),
        "median_latency_span_us": round(max(medians, default=0.0) - min(medians, default=0.0), 6),
        "all_exact_values_ok": value_ok,
    }


def _baseline_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"tokens": 0, "rows": []}
    max_tokens = max(int(row["tokens"]) for row in rows)
    selected = [row for row in rows if int(row["tokens"]) == max_tokens]
    return {
        "tokens": max_tokens,
        "rows": selected,
        "included": [str(row["method"]) for row in selected],
        "scope": "memory-model comparison; quality is reported separately where measured",
    }


def _write_charts(payload: dict[str, Any], chart_dir: Path) -> None:
    (chart_dir / "fixed_state_memory.svg").write_text(_fixed_state_memory_chart(payload["fixed_state_scaling"]))
    (chart_dir / "quality_perplexity.svg").write_text(_quality_perplexity_chart(payload["tinyllama_effective_2m"]))
    (chart_dir / "retrieval_quality.svg").write_text(
        _retrieval_quality_chart(payload["million_token_secret_retrieval"], payload["hkvc_query_scaling"])
    )
    (chart_dir / "baseline_comparison.svg").write_text(_baseline_comparison_chart(payload["modern_baselines"]))


def _fixed_state_memory_chart(rows: list[dict[str, Any]]) -> str:
    max_log = max(math.log10(row["standard_kv_bytes"]) for row in rows)
    body = []
    for index, row in enumerate(rows):
        x = 90 + index * 145
        standard_h = 250 * math.log10(row["standard_kv_bytes"]) / max_log
        state_h = max(3, 250 * math.log10(row["catalyst_state_bytes"]) / max_log)
        body.append(_bar(x, 330 - standard_h, 46, standard_h, "#334155", _format_bytes(row["standard_kv_bytes"])))
        body.append(_bar(x + 56, 330 - state_h, 46, state_h, "#0f766e", _format_bytes(row["catalyst_state_bytes"])))
        body.append(f'<text x="{x + 52}" y="362" text-anchor="middle" font-size="12" fill="#111827">{row["tokens"] // 1000}K</text>')
    return _svg(
        720,
        430,
        "Fixed-State Scaling Guard",
        "Standard KV grows with tokens; Catalyst state remains fixed at the published state boundary.",
        "".join(body) + _legend(92, 390, [("Standard KV", "#334155"), ("Catalyst state", "#0f766e")]),
    )


def _quality_perplexity_chart(row: dict[str, Any]) -> str:
    without = float(row.get("target_ppl_without_recall", 0.0))
    with_recall = float(row.get("target_ppl_with_recall", 0.0))
    max_value = max(without, with_recall, 1.0)
    body = ""
    for x, value, color, label in (
        (180, without, "#7c2d12", "without recall"),
        (340, with_recall, "#2563eb", "with recall"),
    ):
        height = 250 * value / max_value
        body += _bar(x, 330 - height, 74, height, color, f"{value:.2f}")
        body += f'<text x="{x + 37}" y="362" text-anchor="middle" font-size="12" fill="#111827">{label}</text>'
    body += f'<text x="470" y="190" font-size="18" font-weight="700" fill="#0f766e">{float(row.get("ppl_improvement_x", 0.0)):.2f}x better</text>'
    return _svg(
        720,
        430,
        "TinyLlama Effective 2M Context",
        "Target perplexity when Catalyst recall is materialized into the active prompt.",
        body,
    )


def _retrieval_quality_chart(secret: dict[str, Any], scaling: dict[str, Any]) -> str:
    accuracy = float(secret.get("retrieval_accuracy_pct", 0.0))
    latency_span = float(scaling.get("median_latency_span_us", 0.0))
    body = _bar(155, 330 - 250 * accuracy / 100.0, 84, 250 * accuracy / 100.0, "#0f766e", f"{accuracy:.1f}%")
    body += '<text x="197" y="362" text-anchor="middle" font-size="12" fill="#111827">1M retrieval</text>'
    span_height = max(3, min(250, latency_span * 100))
    body += _bar(345, 330 - span_height, 84, span_height, "#7c3aed", f"{latency_span:.3f} us")
    body += '<text x="387" y="362" text-anchor="middle" font-size="12" fill="#111827">latency span</text>'
    body += f'<text x="485" y="190" font-size="15" fill="#111827">largest exact-key run: {scaling.get("largest_entries", 0)}</text>'
    return _svg(
        720,
        430,
        "Retrieval Quality And O(1) Guard",
        "Accuracy plus exact-key latency span across stored-entry scaling.",
        body,
    )


def _baseline_comparison_chart(baselines: dict[str, Any]) -> str:
    rows = baselines.get("rows", [])
    if not rows:
        return _svg(720, 430, "Modern Baseline Comparison", "No baseline rows available.", "")
    max_log = max(math.log10(float(row["memory_mb"])) for row in rows)
    body = []
    for index, row in enumerate(rows):
        x = 70 + index * 128
        height = max(3, 250 * math.log10(float(row["memory_mb"])) / max_log)
        color = "#0f766e" if "Catalyst" in str(row["method"]) else "#475569"
        label = str(row["method"]).replace(" KV cache", "").replace(" Brain HKVC", "")
        body.append(_bar(x, 330 - height, 66, height, color, f'{float(row["memory_mb"]):.4g} MB'))
        body.append(f'<text x="{x + 33}" y="362" text-anchor="middle" font-size="10" fill="#111827">{label[:15]}</text>')
    return _svg(
        760,
        430,
        "Modern Baselines At Largest Published Context",
        "Memory-model comparison against KIVI, TurboQuant, and PyramidKV-style retention.",
        "".join(body),
    )


def _bar(x: float, y: float, width: float, height: float, color: str, label: str) -> str:
    return (
        f'<rect x="{x}" y="{y:.2f}" width="{width}" height="{height:.2f}" rx="5" fill="{color}"/>'
        f'<text x="{x + width / 2}" y="{max(18, y - 8):.2f}" text-anchor="middle" font-size="11" fill="#111827">{label}</text>'
    )


def _legend(x: int, y: int, items: list[tuple[str, str]]) -> str:
    out = []
    offset = 0
    for label, color in items:
        out.append(f'<rect x="{x + offset}" y="{y}" width="14" height="14" fill="{color}"/>')
        out.append(f'<text x="{x + offset + 20}" y="{y + 12}" font-size="12" fill="#111827">{label}</text>')
        offset += 150
    return "".join(out)


def _svg(width: int, height: int, title: str, subtitle: str, body: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f'<text x="56" y="44" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>'
        f'<text x="56" y="70" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">{subtitle}</text>'
        '<line x1="70" y1="330" x2="660" y2="330" stroke="#d1d5db"/>'
        f'{body}</svg>\n'
    )


def _format_bytes(value: float) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    current = float(value)
    for unit in units:
        if current < 1024 or unit == units[-1]:
            return f"{current:.1f} {unit}"
        current /= 1024
    return f"{current:.1f} TiB"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish fixed-state HKVC quality evidence.")
    parser.add_argument("--source-results", default=str(DEFAULT_SOURCE_RESULTS))
    parser.add_argument("--output", default="site/fixed_state_quality_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    args = parser.parse_args()
    result = run_fixed_state_quality_evidence(
        source_results=args.source_results,
        output=args.output,
        chart_dir=args.chart_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
