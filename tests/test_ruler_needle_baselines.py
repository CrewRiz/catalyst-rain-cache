from __future__ import annotations

import json
import sys
from pathlib import Path


EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "evidence" / "hkvc-first-evidence-package"
if str(EVIDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_ROOT))


def test_sliding_window_has_recent_retrieval_positive_control(tmp_path):
    from bench.ruler_needle_benchmark import run_ruler_needle_benchmark

    payload = run_ruler_needle_benchmark(
        output=tmp_path / "ruler.json",
        chart_dir=tmp_path / "charts",
        trials=2,
        distractors=4,
    )

    recent = next(row for row in payload["scenarios"] if row["name"] == "recent_window_positive_control")
    aggregate = payload["aggregate"]

    assert recent["task_type"] == "recent_retrieval"
    assert recent["logical_gap_tokens"] < payload["system_info"]["active_window_tokens"]
    assert recent["catalyst_accuracy_pct"] == 100.0
    assert recent["sliding_window_accuracy_pct"] == 100.0
    assert aggregate["sliding_window_recent_window_accuracy_pct"] == 100.0
    assert aggregate["sliding_window_far_context_retrieval_accuracy_pct"] == 0.0


def test_published_ruler_artifact_labels_far_and_recent_sliding_window():
    written = json.loads((EVIDENCE_ROOT / "site" / "ruler_needle_results.json").read_text())
    recent = next(row for row in written["scenarios"] if row["name"] == "recent_window_positive_control")
    aggregate = written["aggregate"]

    assert aggregate["sliding_window_far_context_retrieval_accuracy_pct"] == 0.0
    assert aggregate["sliding_window_recent_window_accuracy_pct"] == 100.0
    assert aggregate["recent_window_positive_control_passed"] is True
    assert recent["sliding_window_accuracy_pct"] == 100.0
    assert recent["logical_gap_tokens"] < written["system_info"]["active_window_tokens"]
