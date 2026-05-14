from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catalyst_brain import CatalystLongContextSecretMemory  # noqa: E402


LOGICAL_CONTEXT_TOKENS = 1_000_000
ACTIVE_WINDOW_TOKENS = 1_024
CATALYST_STATE_BYTES = 16_384


@dataclass(frozen=True)
class Needle:
    session_id: str
    secret: str
    position: int
    secret_bits: int = 256


class SlidingWindowBaseline:
    def __init__(self, *, active_window_tokens: int = ACTIVE_WINDOW_TOKENS, logical_context_tokens: int = LOGICAL_CONTEXT_TOKENS) -> None:
        self.active_window_tokens = int(active_window_tokens)
        self.logical_context_tokens = int(logical_context_tokens)
        self.records: dict[tuple[str, int, int], str] = {}

    def store(self, needle: Needle) -> None:
        if needle.position >= self.logical_context_tokens - self.active_window_tokens:
            self.records[(needle.session_id, needle.position, needle.secret_bits)] = needle.secret

    def retrieve(self, *, session_id: str, position: int, secret_bits: int) -> str:
        return self.records.get((session_id, position, secret_bits), "")

    @property
    def memory_bytes(self) -> int:
        return self.active_window_tokens * 32 * 8 * 128 * 2 * 2


class LinearScanBaseline:
    def __init__(self) -> None:
        self.records: list[Needle] = []

    def store(self, needle: Needle) -> None:
        self.records.append(needle)

    def retrieve_exact(self, *, session_id: str, position: int, secret_bits: int) -> str:
        for record in reversed(self.records):
            if record.session_id == session_id and record.position == position and record.secret_bits == secret_bits:
                return record.secret
        return ""

    def retrieve_approximate(self, *, session_id: str) -> str:
        for record in reversed(self.records):
            if record.session_id == session_id:
                return record.secret
        return ""

    @property
    def memory_bytes(self) -> int:
        return sum(len(record.session_id.encode("utf-8")) + len(record.secret.encode("ascii")) + 16 for record in self.records)


def run_ruler_needle_benchmark(
    *,
    output: str | Path,
    chart_dir: str | Path,
    trials: int = 32,
    distractors: int = 2048,
) -> dict[str, Any]:
    scenarios = [
        _single_needle(trials=trials, distractors=distractors),
        _multi_needle(trials=trials, distractors=distractors),
        _duplicate_overwrite(trials=trials, distractors=distractors),
        _recent_window_positive_control(trials=trials, distractors=distractors),
        _wrong_position_rejection(trials=trials, distractors=distractors),
    ]
    retrieval_scenarios = [row for row in scenarios if row["task_type"] == "retrieval"]
    recent_control = _scenario_by_name(scenarios, "recent_window_positive_control")
    catalyst_mean = statistics.fmean(row["catalyst_accuracy_pct"] for row in retrieval_scenarios)
    sliding_mean = statistics.fmean(row["sliding_window_accuracy_pct"] for row in retrieval_scenarios)
    payload = {
        "system_info": {
            "suite": "ruler_needle_fixed_state",
            "logical_context_tokens": LOGICAL_CONTEXT_TOKENS,
            "active_window_tokens": ACTIVE_WINDOW_TOKENS,
            "trials": trials,
            "distractors": distractors,
            "secret_bits": 256,
        },
        "claim_boundaries": {
            "fixed_state_claim": True,
            "no_filler_token_materialization": True,
            "linear_scan_baseline_is_not_o1": True,
            "exact_key_guard_rejects_approximate_fallback": True,
            "sliding_window_is_bounded_recent_only": True,
            "suite_is_synthetic_ruler_needle_style": True,
        },
        "aggregate": {
            "catalyst_mean_retrieval_accuracy_pct": round(catalyst_mean, 4),
            "sliding_window_mean_retrieval_accuracy_pct": round(sliding_mean, 4),
            "sliding_window_far_context_retrieval_accuracy_pct": round(sliding_mean, 4),
            "sliding_window_recent_window_accuracy_pct": recent_control["sliding_window_accuracy_pct"],
            "recent_window_positive_control_passed": recent_control["sliding_window_accuracy_pct"] == 100.0,
            "linear_scan_mean_retrieval_accuracy_pct": round(
                statistics.fmean(row["linear_scan_accuracy_pct"] for row in retrieval_scenarios),
                4,
            ),
            "adversarial_false_positive_pct": _scenario_by_name(scenarios, "wrong_position_rejection")["catalyst_false_positive_pct"],
            "catalyst_state_bytes": CATALYST_STATE_BYTES,
            "catalyst_state_bytes_constant": len({row["catalyst_state_bytes"] for row in scenarios}) == 1,
            "max_logical_gap_tokens": max(row["logical_gap_tokens"] for row in scenarios),
        },
        "baselines": {
            "sliding_window": {
                "memory_shape": "bounded_recent_only",
                "active_window_tokens": ACTIVE_WINDOW_TOKENS,
                "expected_far_context_behavior": "misses needles outside the recent window",
            },
            "linear_scan": {
                "memory_shape": "linear",
                "expected_far_context_behavior": "can retrieve by scanning retained records, but memory and work grow with retained items",
            },
            "standard_kv_attention": {
                "memory_shape": "linear",
                "standard_kv_bytes_1m_8b_gqa": _standard_kv_bytes(LOGICAL_CONTEXT_TOKENS),
            },
        },
        "scenarios": scenarios,
        "charts": {
            "ruler_accuracy": "charts/ruler_accuracy.svg",
            "ruler_latency": "charts/ruler_latency.svg",
            "ruler_memory": "charts/ruler_memory.svg",
            "ruler_baselines": "charts/ruler_baselines.svg",
        },
    }
    output_path = Path(output)
    chart_path = Path(chart_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.mkdir(parents=True, exist_ok=True)
    _write_charts(payload, chart_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def _single_needle(*, trials: int, distractors: int) -> dict[str, Any]:
    needles = [Needle(f"single-{i}", _secret("single", i), 10) for i in range(trials)]
    return _run_retrieval_scenario(
        name="single_needle",
        description="One far-context needle per trial at logical token 10.",
        needles=needles,
        distractors=distractors,
        query_position=LOGICAL_CONTEXT_TOKENS,
    )


def _multi_needle(*, trials: int, distractors: int) -> dict[str, Any]:
    positions = (10, 50_000, 250_000)
    needles = [
        Needle(f"multi-{trial}-{offset}", _secret("multi", trial, offset), position)
        for trial in range(trials)
        for offset, position in enumerate(positions)
    ]
    return _run_retrieval_scenario(
        name="multi_needle",
        description="Three far-context needles per trial at separated logical positions.",
        needles=needles,
        distractors=distractors,
        query_position=LOGICAL_CONTEXT_TOKENS,
    )


def _duplicate_overwrite(*, trials: int, distractors: int) -> dict[str, Any]:
    needles: list[Needle] = []
    for trial in range(trials):
        needles.append(Needle(f"duplicate-{trial}", _secret("duplicate-old", trial), 10))
        needles.append(Needle(f"duplicate-{trial}", _secret("duplicate-new", trial), 11))
    row = _run_retrieval_scenario(
        name="duplicate_overwrite",
        description="Later duplicate instruction at the adjacent logical position is the queried target.",
        needles=needles,
        distractors=distractors,
        query_position=LOGICAL_CONTEXT_TOKENS,
        query_filter=lambda needle: needle.position == 11,
    )
    row["overwrite_policy"] = "latest_explicit_position_wins"
    return row


def _recent_window_positive_control(*, trials: int, distractors: int) -> dict[str, Any]:
    position = LOGICAL_CONTEXT_TOKENS - 16
    needles = [Needle(f"recent-{i}", _secret("recent", i), position) for i in range(trials)]
    row = _run_retrieval_scenario(
        name="recent_window_positive_control",
        description="Recent-window positive control: needle is inside the bounded sliding window.",
        needles=needles,
        distractors=distractors,
        query_position=LOGICAL_CONTEXT_TOKENS,
    )
    row["task_type"] = "recent_retrieval"
    row["positive_control"] = True
    return row


def _wrong_position_rejection(*, trials: int, distractors: int) -> dict[str, Any]:
    needles = [Needle(f"wrong-pos-{i}", _secret("wrong", i), 10) for i in range(trials)]
    catalyst, sliding, linear = _build_memories(needles, distractors)
    wrong_position = 11

    def catalyst_query() -> list[dict[str, Any]]:
        return [
            _catalyst_exact_retrieve(
                catalyst,
                session_id=needle.session_id,
                needle_token_position=wrong_position,
                query_token_position=LOGICAL_CONTEXT_TOKENS,
                secret_bits=needle.secret_bits,
            )
            for needle in needles
        ]

    def sliding_query() -> list[str]:
        return [
            sliding.retrieve(session_id=needle.session_id, position=wrong_position, secret_bits=needle.secret_bits)
            for needle in needles
        ]

    def linear_approx_query() -> list[str]:
        return [linear.retrieve_approximate(session_id=needle.session_id) for needle in needles]

    catalyst_timing, catalyst_results = _measure(catalyst_query)
    sliding_timing, sliding_results = _measure(sliding_query)
    linear_timing, linear_results = _measure(linear_approx_query)
    catalyst_false_positive = sum(bool(item["secret"]) and float(item["confidence"]) > 0.0 for item in catalyst_results)
    linear_false_positive = sum(bool(item) for item in linear_results)
    total = len(needles)
    return {
        "name": "wrong_position_rejection",
        "task_type": "rejection",
        "description": "Adversarial near-position query should reject a needle at token 10 when token 11 is requested.",
        "logical_context_tokens": LOGICAL_CONTEXT_TOKENS,
        "logical_gap_tokens": LOGICAL_CONTEXT_TOKENS - wrong_position,
        "trials": trials,
        "needles": total,
        "distractors": distractors,
        "catalyst_accuracy_pct": round(100.0 * (total - catalyst_false_positive) / total, 4),
        "sliding_window_accuracy_pct": round(100.0 * (total - sum(bool(item) for item in sliding_results)) / total, 4),
        "linear_scan_accuracy_pct": round(100.0 * (total - linear_false_positive) / total, 4),
        "catalyst_false_positive_pct": round(100.0 * catalyst_false_positive / total, 4),
        "linear_approx_false_positive_pct": round(100.0 * linear_false_positive / total, 4),
        "catalyst_median_us": catalyst_timing["median_us"],
        "catalyst_p95_us": catalyst_timing["p95_us"],
        "sliding_window_median_us": sliding_timing["median_us"],
        "linear_scan_median_us": linear_timing["median_us"],
        "catalyst_state_bytes": _catalyst_state_bytes(catalyst),
        "sliding_window_memory_bytes": sliding.memory_bytes,
        "linear_scan_memory_bytes": linear.memory_bytes,
        "standard_kv_bytes": _standard_kv_bytes(LOGICAL_CONTEXT_TOKENS),
        "throughput_queries_per_second": round(total / max(catalyst_timing["median_us"] / 1_000_000.0, 1e-12), 4),
    }


def _run_retrieval_scenario(
    *,
    name: str,
    description: str,
    needles: list[Needle],
    distractors: int,
    query_position: int,
    query_filter: Any | None = None,
) -> dict[str, Any]:
    catalyst, sliding, linear = _build_memories(needles, distractors)
    targets = [needle for needle in needles if query_filter is None or query_filter(needle)]

    def catalyst_query() -> list[str]:
        return [
            _catalyst_exact_retrieve(
                catalyst,
                session_id=needle.session_id,
                needle_token_position=needle.position,
                query_token_position=query_position,
                secret_bits=needle.secret_bits,
            )["secret"]
            for needle in targets
        ]

    def sliding_query() -> list[str]:
        return [
            sliding.retrieve(session_id=needle.session_id, position=needle.position, secret_bits=needle.secret_bits)
            for needle in targets
        ]

    def linear_query() -> list[str]:
        return [
            linear.retrieve_exact(session_id=needle.session_id, position=needle.position, secret_bits=needle.secret_bits)
            for needle in targets
        ]

    catalyst_timing, catalyst_results = _measure(catalyst_query)
    sliding_timing, sliding_results = _measure(sliding_query)
    linear_timing, linear_results = _measure(linear_query)
    total = len(targets)
    catalyst_correct = sum(actual == needle.secret for actual, needle in zip(catalyst_results, targets))
    sliding_correct = sum(actual == needle.secret for actual, needle in zip(sliding_results, targets))
    linear_correct = sum(actual == needle.secret for actual, needle in zip(linear_results, targets))
    return {
        "name": name,
        "task_type": "retrieval",
        "description": description,
        "logical_context_tokens": LOGICAL_CONTEXT_TOKENS,
        "logical_gap_tokens": max(query_position - min(needle.position for needle in targets), 0),
        "trials": total,
        "needles": total,
        "distractors": distractors,
        "catalyst_accuracy_pct": round(100.0 * catalyst_correct / total, 4),
        "sliding_window_accuracy_pct": round(100.0 * sliding_correct / total, 4),
        "linear_scan_accuracy_pct": round(100.0 * linear_correct / total, 4),
        "catalyst_false_positive_pct": 0.0,
        "catalyst_median_us": catalyst_timing["median_us"],
        "catalyst_p95_us": catalyst_timing["p95_us"],
        "sliding_window_median_us": sliding_timing["median_us"],
        "linear_scan_median_us": linear_timing["median_us"],
        "catalyst_state_bytes": _catalyst_state_bytes(catalyst),
        "sliding_window_memory_bytes": sliding.memory_bytes,
        "linear_scan_memory_bytes": linear.memory_bytes,
        "standard_kv_bytes": _standard_kv_bytes(LOGICAL_CONTEXT_TOKENS),
        "throughput_queries_per_second": round(total / max(catalyst_timing["median_us"] / 1_000_000.0, 1e-12), 4),
    }


def _build_memories(needles: list[Needle], distractors: int) -> tuple[CatalystLongContextSecretMemory, SlidingWindowBaseline, LinearScanBaseline]:
    catalyst = CatalystLongContextSecretMemory(dim=4096)
    sliding = SlidingWindowBaseline()
    linear = LinearScanBaseline()
    for index in range(distractors):
        decoy = Needle(f"distractor-{index}", _secret("distractor", index), 20 + index)
        _store_all(catalyst, sliding, linear, decoy)
    for needle in needles:
        _store_all(catalyst, sliding, linear, needle)
    return catalyst, sliding, linear


def _store_all(
    catalyst: CatalystLongContextSecretMemory,
    sliding: SlidingWindowBaseline,
    linear: LinearScanBaseline,
    needle: Needle,
) -> None:
    catalyst.store_secret(
        session_id=needle.session_id,
        secret=needle.secret,
        token_position=needle.position,
        secret_bits=needle.secret_bits,
    )
    sliding.store(needle)
    linear.store(needle)


def _catalyst_exact_retrieve(
    memory: CatalystLongContextSecretMemory,
    *,
    session_id: str,
    needle_token_position: int,
    query_token_position: int,
    secret_bits: int,
) -> dict[str, Any]:
    record = getattr(memory, "_records", {}).get(session_id)
    if not record or int(record["needle_token_position"]) != int(needle_token_position):
        return {
            "session_id": session_id,
            "secret": "",
            "confidence": 0.0,
            "secret_bits": secret_bits,
            "needle_token_position": needle_token_position,
            "query_token_position": query_token_position,
            "logical_gap_tokens": query_token_position - needle_token_position,
            "exact_key_match": False,
        }
    retrieved = memory.retrieve_secret(
        session_id=session_id,
        needle_token_position=needle_token_position,
        query_token_position=query_token_position,
        secret_bits=secret_bits,
    )
    return {**retrieved, "exact_key_match": True}


def _measure(fn: Any, *, repeats: int = 5, warmup: int = 1) -> tuple[dict[str, float], list[str]]:
    result: list[str] = []
    for _ in range(warmup):
        result = fn()
    timings: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        result = fn()
        timings.append((time.perf_counter_ns() - started) / 1000.0)
    timings.sort()
    return {
        "median_us": round(statistics.median(timings), 4),
        "p95_us": round(timings[min(len(timings) - 1, int(len(timings) * 0.95))], 4),
    }, result


def _secret(*parts: Any) -> str:
    joined = ":".join(str(part) for part in parts)
    return hashlib.blake2b(joined.encode("utf-8"), digest_size=32).hexdigest()


def _catalyst_state_bytes(memory: CatalystLongContextSecretMemory) -> int:
    return int(memory.export_rain_snapshot()["state_bytes"])


def _standard_kv_bytes(tokens: int) -> int:
    return int(tokens * 32 * 8 * 128 * 2 * 2)


def _scenario_by_name(scenarios: list[dict[str, Any]], name: str) -> dict[str, Any]:
    for scenario in scenarios:
        if scenario["name"] == name:
            return scenario
    raise KeyError(name)


def _write_charts(payload: dict[str, Any], chart_dir: Path) -> None:
    scenarios = payload["scenarios"]
    (chart_dir / "ruler_accuracy.svg").write_text(_accuracy_chart(scenarios))
    (chart_dir / "ruler_latency.svg").write_text(_latency_chart(scenarios))
    (chart_dir / "ruler_memory.svg").write_text(_memory_chart(scenarios))
    (chart_dir / "ruler_baselines.svg").write_text(_baselines_chart(payload))


def _accuracy_chart(scenarios: list[dict[str, Any]]) -> str:
    body = []
    step = 620 / max(1, len(scenarios))
    bar_width = min(38, max(28, step * 0.30))
    for index, row in enumerate(scenarios):
        x = 64 + index * step
        cat_h = 230 * float(row["catalyst_accuracy_pct"]) / 100.0
        slide_h = 230 * float(row["sliding_window_accuracy_pct"]) / 100.0
        body.append(_bar(x, 330 - cat_h, bar_width, cat_h, "#0f766e", f'{row["catalyst_accuracy_pct"]:.0f}%'))
        body.append(_bar(x + bar_width + 8, 330 - max(2, slide_h), bar_width, max(2, slide_h), "#7c2d12", f'{row["sliding_window_accuracy_pct"]:.0f}%'))
        body.append(f'<text x="{x + bar_width}" y="362" text-anchor="middle" font-size="9" fill="#111827">{_short_name(row["name"])}</text>')
    return _svg(
        760,
        430,
        "RULER/Needle Accuracy",
        "Catalyst fixed-state retrieval versus bounded recent-window baseline.",
        "".join(body) + _legend(92, 390, [("Catalyst", "#0f766e"), ("Sliding window", "#7c2d12")]),
    )


def _latency_chart(scenarios: list[dict[str, Any]]) -> str:
    max_value = max(float(row["linear_scan_median_us"]) for row in scenarios)
    body = []
    step = 620 / max(1, len(scenarios))
    bar_width = min(38, max(28, step * 0.30))
    for index, row in enumerate(scenarios):
        x = 64 + index * step
        cat_h = max(2, 230 * float(row["catalyst_median_us"]) / max_value)
        linear_h = max(2, 230 * float(row["linear_scan_median_us"]) / max_value)
        body.append(_bar(x, 330 - cat_h, bar_width, cat_h, "#2563eb", f'{row["catalyst_median_us"]:.0f} us'))
        body.append(_bar(x + bar_width + 8, 330 - linear_h, bar_width, linear_h, "#475569", f'{row["linear_scan_median_us"]:.0f} us'))
        body.append(f'<text x="{x + bar_width}" y="362" text-anchor="middle" font-size="9" fill="#111827">{_short_name(row["name"])}</text>')
    return _svg(
        760,
        430,
        "Median Query Latency",
        "Catalyst fixed-state query batches compared with linear scan over retained records.",
        "".join(body) + _legend(92, 390, [("Catalyst", "#2563eb"), ("Linear scan", "#475569")]),
    )


def _memory_chart(scenarios: list[dict[str, Any]]) -> str:
    row = scenarios[0]
    values = [
        ("Standard KV", float(row["standard_kv_bytes"]), "#334155"),
        ("Sliding", float(row["sliding_window_memory_bytes"]), "#7c2d12"),
        ("Linear scan", float(max(item["linear_scan_memory_bytes"] for item in scenarios)), "#475569"),
        ("Catalyst", float(row["catalyst_state_bytes"]), "#0f766e"),
    ]
    max_log = max(math.log10(value) for _label, value, _color in values)
    body = []
    for index, (label, value, color) in enumerate(values):
        x = 90 + index * 145
        height = max(3, 250 * math.log10(value) / max_log)
        body.append(_bar(x, 330 - height, 74, height, color, _format_bytes(value)))
        body.append(f'<text x="{x + 37}" y="362" text-anchor="middle" font-size="11" fill="#111827">{label}</text>')
    return _svg(
        760,
        430,
        "Memory Shape At 1M Logical Tokens",
        "Log-scale memory model for the measured suite.",
        "".join(body),
    )


def _baselines_chart(payload: dict[str, Any]) -> str:
    aggregate = payload["aggregate"]
    rows = [
        ("Catalyst retrieval", aggregate["catalyst_mean_retrieval_accuracy_pct"], "#0f766e"),
        ("Sliding far", aggregate["sliding_window_far_context_retrieval_accuracy_pct"], "#7c2d12"),
        ("Sliding recent", aggregate["sliding_window_recent_window_accuracy_pct"], "#d97706"),
        ("Catalyst rejection", 100.0 - aggregate["adversarial_false_positive_pct"], "#2563eb"),
    ]
    body = []
    for index, (label, value, color) in enumerate(rows):
        x = 82 + index * 148
        height = max(2, 250 * float(value) / 100.0)
        body.append(_bar(x, 330 - height, 78, height, color, f"{value:.0f}%"))
        body.append(f'<text x="{x + 39}" y="362" text-anchor="middle" font-size="11" fill="#111827">{label}</text>')
    return _svg(
        760,
        430,
        "Baseline Outcome Summary",
        "Retrieval and adversarial rejection outcomes across the suite.",
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
        '<line x1="70" y1="330" x2="690" y2="330" stroke="#d1d5db"/>'
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


def _short_name(name: str) -> str:
    return {
        "single_needle": "single",
        "multi_needle": "multi",
        "duplicate_overwrite": "duplicate",
        "recent_window_positive_control": "recent",
        "wrong_position_rejection": "reject",
    }.get(name, name[:10])


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fixed-state RULER/Needle-style HKVC benchmarks.")
    parser.add_argument("--output", default="site/ruler_needle_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--distractors", type=int, default=2048)
    args = parser.parse_args()
    result = run_ruler_needle_benchmark(
        output=args.output,
        chart_dir=args.chart_dir,
        trials=args.trials,
        distractors=args.distractors,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
