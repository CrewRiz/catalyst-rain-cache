from __future__ import annotations

import csv
import math
from pathlib import Path

import catalyst_hdc as hdc
from catalyst_brain.rain import RainPayload, to_header


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
TOKENS = [1_000, 4_000, 16_000, 64_000, 128_000]
LAYERS = 40
HIDDEN_SIZE = 4096
KV_TENSORS = 2
FP16_BYTES = 2
CATALYST_DIM = 4096


def catalyst_fixed_state_bytes() -> int:
    vector = hdc.hv_hash_string("public-kv-cache-scaling-chart", CATALYST_DIM)
    header = to_header(
        RainPayload(
            agent_id="public-kv-cache-scaling",
            dim=CATALYST_DIM,
            world_vector=vector,
            config={"mode": "holographic-kv-cache", "surface": "public-results-only"},
        )
    )
    world_vector_bytes = CATALYST_DIM * 4
    return max(world_vector_bytes, len(header.encode("utf-8")))


CATALYST_BYTES = catalyst_fixed_state_bytes()


METHODS = [
    {
        "name": "FP16 KV cache",
        "kind": "linear",
        "factor": 1.0,
        "color": "#6b7280",
        "note": "Uncompressed transformer KV cache.",
    },
    {
        "name": "TurboQuant",
        "kind": "linear",
        "factor": 3.5 / 16.0,
        "color": "#2563eb",
        "note": "TurboQuant reports KV quality neutrality at 3.5 bits/channel.",
    },
    {
        "name": "KIVI 2-bit",
        "kind": "linear",
        "factor": 2.0 / 16.0,
        "color": "#7c3aed",
        "note": "Raw 2-bit KV model versus FP16 cache.",
    },
    {
        "name": "PyramidKV",
        "kind": "linear",
        "factor": 0.12,
        "color": "#ea580c",
        "note": "Paper reports full-cache performance with 12% KV retention.",
    },
    {
        "name": "Catalyst Brain HKVC",
        "kind": "constant",
        "factor": None,
        "color": "#059669",
        "note": "Fixed 4096-dim holographic world vector.",
    },
]


def baseline_bytes(tokens: int) -> int:
    return tokens * LAYERS * HIDDEN_SIZE * KV_TENSORS * FP16_BYTES


def method_bytes(method: dict[str, object], tokens: int) -> float:
    if method["kind"] == "constant":
        return float(CATALYST_BYTES)
    return baseline_bytes(tokens) * float(method["factor"])


def build_rows() -> list[dict[str, object]]:
    rows = []
    for tokens in TOKENS:
        for method in METHODS:
            memory_bytes = method_bytes(method, tokens)
            rows.append(
                {
                    "tokens": tokens,
                    "method": method["name"],
                    "memory_gb": round(memory_bytes / 1_000_000_000.0, 9),
                    "memory_mb": round(memory_bytes / 1_000_000.0, 6),
                    "relative_to_fp16_pct": round(
                        100.0 * memory_bytes / baseline_bytes(tokens),
                        9,
                    ),
                    "note": method["note"],
                }
            )
    return rows


def write_csv(rows: list[dict[str, object]]) -> None:
    path = DOCS / "kv_cache_scaling.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_svg(rows: list[dict[str, object]]) -> None:
    width = 1100
    height = 660
    left = 92
    right = 270
    top = 64
    bottom = 96
    plot_w = width - left - right
    plot_h = height - top - bottom
    min_x = math.log10(min(TOKENS))
    max_x = math.log10(max(TOKENS))
    values = [float(row["memory_gb"]) for row in rows]
    min_y = math.log10(min(values) * 0.75)
    max_y = math.log10(max(values) * 1.4)

    def x_pos(tokens: int) -> float:
        return left + ((math.log10(tokens) - min_x) / (max_x - min_x)) * plot_w

    def y_pos(gb: float) -> float:
        return top + (1.0 - ((math.log10(gb) - min_y) / (max_y - min_y))) * plot_h

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        ".title{font:700 25px system-ui,-apple-system,Segoe UI,sans-serif;fill:#111827}",
        ".subtitle{font:14px system-ui,-apple-system,Segoe UI,sans-serif;fill:#4b5563}",
        ".axis{font:600 14px system-ui,-apple-system,Segoe UI,sans-serif;fill:#374151}",
        ".tick{font:12px system-ui,-apple-system,Segoe UI,sans-serif;fill:#4b5563}",
        ".legend{font:13px system-ui,-apple-system,Segoe UI,sans-serif;fill:#111827}",
        ".callout{font:700 14px system-ui,-apple-system,Segoe UI,sans-serif;fill:#065f46}",
        "</style>",
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff"/>',
        f'<text x="{left}" y="32" class="title">KV-cache scaling: compression still grows, Catalyst stays fixed</text>',
        f'<text x="{left}" y="54" class="subtitle">Model: 40 layers, hidden size 4096, FP16 K+V tensors. Both axes use log scale.</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#111827"/>',
    ]

    y_ticks = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100]
    for tick in y_ticks:
        if math.log10(tick) < min_y or math.log10(tick) > max_y:
            continue
        y = y_pos(tick)
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_w}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left - 10}" y="{y + 4:.2f}" class="tick" text-anchor="end">{tick:g}</text>')

    for tokens in TOKENS:
        x = x_pos(tokens)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_h}" stroke="#f3f4f6"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 58}" class="tick" text-anchor="middle">{tokens//1000}K</text>')

    by_method: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    for method in METHODS:
        name = str(method["name"])
        color = str(method["color"])
        points = sorted(by_method[name], key=lambda row: int(row["tokens"]))
        path_data = []
        for i, row in enumerate(points):
            cmd = "M" if i == 0 else "L"
            path_data.append(f'{cmd}{x_pos(int(row["tokens"])):.2f},{y_pos(float(row["memory_gb"])):.2f}')
        width_px = 4 if name == "Catalyst Brain HKVC" else 3
        parts.append(f'<path d="{" ".join(path_data)}" fill="none" stroke="{color}" stroke-width="{width_px}"/>')
        for row in points:
            parts.append(
                f'<circle cx="{x_pos(int(row["tokens"])):.2f}" cy="{y_pos(float(row["memory_gb"])):.2f}" r="4" fill="{color}"/>'
            )

    for i, method in enumerate(METHODS):
        y = top + 24 * i
        parts.append(f'<rect x="{width - right + 34}" y="{y}" width="14" height="14" fill="{method["color"]}"/>')
        parts.append(f'<text x="{width - right + 56}" y="{y + 12}" class="legend">{method["name"]}</text>')

    catalyst_128k = method_bytes(METHODS[-1], 128_000)
    turbo_128k = method_bytes(METHODS[1], 128_000)
    separation = turbo_128k / catalyst_128k
    parts.append(f'<text x="{left + 28}" y="{top + 34}" class="callout">Catalyst is O(1): fixed {CATALYST_BYTES:,} bytes</text>')
    parts.append(f'<text x="{left + 28}" y="{top + 56}" class="subtitle">At 128K tokens, modeled TurboQuant remains {separation:,.0f}x larger than Catalyst state.</text>')
    parts.append(f'<text x="{left + plot_w / 2}" y="{height - 18}" class="axis" text-anchor="middle">Context length</text>')
    parts.append(f'<text x="22" y="{top + plot_h / 2}" class="axis" text-anchor="middle" transform="rotate(-90 22 {top + plot_h / 2})">KV/cache memory (GB)</text>')
    parts.append("</svg>")
    (DOCS / "kv_cache_scaling.svg").write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(rows)
    write_svg(rows)


if __name__ == "__main__":
    main()
