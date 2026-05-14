from __future__ import annotations

import json
import sys
from pathlib import Path


EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "evidence" / "hkvc-first-evidence-package"
if str(EVIDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_ROOT))


def test_rain_transport_probe_writes_payload_and_chart(tmp_path):
    from bench.rain_transport_probe import run_rain_transport_probe

    items = [
        {
            "_id": "lb-1",
            "domain": "Single-Document QA",
            "sub_domain": "unit",
            "difficulty": "easy",
            "length": "long",
            "question": "Which option is correct?",
            "choice_A": "needle-42",
            "choice_B": "wrong",
            "choice_C": "wrong",
            "choice_D": "wrong",
            "answer": "A",
            "context": "needle-42 " * 5000,
        }
    ]

    payload = run_rain_transport_probe(
        output=tmp_path / "transport.json",
        chart_dir=tmp_path / "charts",
        longbench_items=items,
        longbench_limit=1,
    )

    written = json.loads((tmp_path / "transport.json").read_text())
    assert payload["claim_boundaries"]["transport_only_not_model_quality"] is True
    assert written["aggregate"]["sample_count"] == 1
    assert written["aggregate"]["mean_packed_vs_raw_json_ratio"] < 1.0
    assert written["rows"][0]["local_truncation_disabled"] is True
    assert (tmp_path / written["charts"]["rain_transport_payload"]).exists()


def test_published_rain_transport_probe_schema():
    output = EVIDENCE_ROOT / "site" / "rain_transport_probe_results.json"

    if not output.exists():
        return
    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "rain_transport_probe"
    assert written["claim_boundaries"]["packed_token_transport_is_not_model_context_extension"] is True
    assert written["aggregate"]["sample_count"] >= 1
