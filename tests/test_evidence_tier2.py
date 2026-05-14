from __future__ import annotations

import json
import sys
from pathlib import Path


EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "evidence" / "hkvc-first-evidence-package"
if str(EVIDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_ROOT))


def test_tier2_evidence_package_schema_from_published_artifact():
    output = EVIDENCE_ROOT / "site" / "tier2_evidence_results.json"

    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "hkvc_tier2_evidence"
    assert written["claim_boundaries"]["official_longbench"] is False
    assert written["claim_boundaries"]["baselines_are_modeled_unless_marked_measured"] is True
    assert len(written["model_level_perplexity"]["corpora"]) >= 4
    assert written["model_level_perplexity"]["status"] == "measured"
    assert written["model_level_perplexity"]["model_name"] == "distilgpt2"
    assert written["model_level_perplexity"]["aggregate"]["catalyst_ppl_improvement_x"] > 1.0
    assert written["extended_longbench_ruler"]["aggregate"]["task_count"] >= 5
    assert written["extended_longbench_ruler"]["aggregate"]["catalyst_accuracy_pct"] >= 80.0
    assert written["generation_quality"]["aggregate"]["mean_rubric_score"] >= 0.75
    baseline_names = {row["method"] for row in written["baseline_efficiency"]["rows"]}
    assert {
        "FP16 KV",
        "KIVI 2-bit",
        "TurboQuant 3.5-bit",
        "PyramidKV 12%",
        "H2O 20%",
        "vLLM PagedAttention",
        "Sparse/Retrieval Attention",
        "Catalyst ManifoldAttention",
    }.issubset(baseline_names)
    for chart in written["charts"].values():
        assert (EVIDENCE_ROOT / "site" / chart).exists()
    assert written["baseline_efficiency"]["aggregate"]["catalyst_memory_rank"] == 1


def test_tier2_runner_delegates_to_catalyst_brain(monkeypatch, tmp_path):
    import bench.tier2_evidence as tier2

    calls = {}

    def fake_call(names, **kwargs):
        calls["names"] = names
        calls["kwargs"] = kwargs
        return {"delegated": True}

    monkeypatch.setattr(tier2, "call_catalyst_brain", fake_call)

    payload = tier2.run_tier2_evidence(
        output=tmp_path / "tier2.json",
        chart_dir=tmp_path / "charts",
        run_model=False,
    )

    assert payload == {"delegated": True}
    assert "run_hkvc_tier2_evidence" in calls["names"]
    assert calls["kwargs"]["run_model"] is False


def test_lossless_equivalence_package_schema_from_published_artifact():
    output = EVIDENCE_ROOT / "site" / "lossless_equivalence_results.json"

    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "hkvc_lossless_equivalence"
    assert written["claim_boundaries"]["universal_lossless_requires_exact_state"] is True
    assert written["claim_boundaries"]["not_fixed_size_lossless_compression"] is True
    assert written["aggregate"]["status"] == "measured"
    assert written["aggregate"]["allclose_pct"] == 100.0
    assert written["aggregate"]["max_abs_error"] <= 1e-5
    assert written["aggregate"]["max_local_window_tokens"] < written["aggregate"]["max_logical_tokens"]
    assert written["aggregate"]["scenario_count"] >= 3
    assert {
        "decode_overflow",
        "causal_prefill",
        "additive_mask",
    }.issubset(set(written["aggregate"]["scenario_modes"]))
    for chart in written["charts"].values():
        assert (EVIDENCE_ROOT / "site" / chart).exists()


def test_lossless_scale_package_schema_from_published_artifact():
    output = EVIDENCE_ROOT / "site" / "lossless_scale_results.json"

    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "hkvc_lossless_scale"
    assert written["claim_boundaries"]["operator_level_not_full_model_generation"] is True
    assert written["claim_boundaries"]["universal_lossless_requires_exact_state"] is True
    assert written["claim_boundaries"]["not_fixed_size_lossless_compression"] is True
    assert written["aggregate"]["status"] == "measured"
    assert written["aggregate"]["profile_count"] >= 4
    assert written["aggregate"]["allclose_pct"] == 100.0
    assert written["aggregate"]["max_abs_error"] <= 1e-5
    assert written["aggregate"]["max_local_window_tokens"] < written["aggregate"]["max_logical_tokens"]
    assert written["aggregate"]["max_layers"] >= 3
    assert {
        "decode_overflow",
        "causal_prefill",
        "additive_mask",
        "layer_stack_decode",
    }.issubset(set(written["aggregate"]["profile_modes"]))
    for chart in written["charts"].values():
        assert (EVIDENCE_ROOT / "site" / chart).exists()


def test_lossless_runner_delegates_to_catalyst_brain(monkeypatch, tmp_path):
    import bench.lossless_equivalence as lossless

    calls = {}

    def fake_call(names, **kwargs):
        calls["names"] = names
        calls["kwargs"] = kwargs
        return {"delegated": True}

    monkeypatch.setattr(lossless, "call_catalyst_brain", fake_call)

    payload = lossless.run_lossless_equivalence(
        output=tmp_path / "lossless.json",
        chart_dir=tmp_path / "charts",
        trials=2,
    )

    assert payload == {"delegated": True}
    assert "run_hkvc_lossless_equivalence_benchmark" in calls["names"]
    assert calls["kwargs"]["trials"] == 2


def test_lossless_scale_runner_delegates_to_catalyst_brain(monkeypatch, tmp_path):
    import bench.lossless_scale as lossless_scale

    calls = {}

    def fake_call(names, **kwargs):
        calls["names"] = names
        calls["kwargs"] = kwargs
        return {"delegated": True}

    monkeypatch.setattr(lossless_scale, "call_catalyst_brain", fake_call)

    payload = lossless_scale.run_lossless_scale(
        output=tmp_path / "lossless_scale.json",
        chart_dir=tmp_path / "charts",
        trials=2,
    )

    assert payload == {"delegated": True}
    assert "run_hkvc_lossless_scale_benchmark" in calls["names"]
    assert calls["kwargs"]["trials"] == 2


def test_next_evidence_package_schema_from_published_artifact():
    output = EVIDENCE_ROOT / "site" / "next_evidence_results.json"

    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "next_evidence_tier"
    assert written["claim_boundaries"]["official_scores_not_yet_claimed"] is True
    assert written["official_benchmark_readiness"]["aggregate"]["tracked_benchmark_count"] >= 2
    tracked = {row["name"] for row in written["official_benchmark_readiness"]["rows"]}
    assert {"LongBench", "RULER"}.issubset(tracked)
    assert written["public_adapter_latency"]["aggregate"]["updates"] >= 128
    assert written["public_adapter_latency"]["aggregate"]["passthrough_update_p50_us"] > 0
    assert written["public_adapter_latency"]["aggregate"]["refs_update_p50_us"] > 0
    assert written["baseline_import_readiness"]["aggregate"]["tracked_baseline_count"] >= 5
    assert written["cloudflare_workers_ai"]["status"] in {"measured", "skipped_missing_credentials"}
    if written["cloudflare_workers_ai"]["status"] == "measured":
        assert written["cloudflare_workers_ai"]["quality_probe"]["score"] == 1.0
    else:
        assert written["cloudflare_workers_ai"]["quality_probe"]["ran"] is False
        assert written["cloudflare_workers_ai"]["api_token_present"] is False
    assert "70b" in written["cloudflare_workers_ai"]["model"].lower()
    assert written["remaining_blockers"]
    for chart in written["charts"].values():
        assert (EVIDENCE_ROOT / "site" / chart).exists()


def test_next_evidence_runner_writes_artifacts(monkeypatch, tmp_path):
    import bench.next_evidence as next_evidence

    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "test-token")
    payload = next_evidence.run_next_evidence(
        output=tmp_path / "next.json",
        chart_dir=tmp_path / "charts",
        updates=128,
        network=False,
    )

    assert payload["system_info"]["suite"] == "next_evidence_tier"
    assert payload["official_benchmark_readiness"]["aggregate"]["tracked_benchmark_count"] >= 2
    assert payload["cloudflare_workers_ai"]["status"] == "skipped_network_disabled"
    assert payload["cloudflare_workers_ai"]["quality_probe"]["ran"] is False
    assert payload["public_adapter_latency"]["aggregate"]["updates"] == 128
    assert (tmp_path / "charts" / "next_official_readiness.svg").exists()
    assert (tmp_path / "charts" / "next_adapter_latency.svg").exists()


def test_generation_rubric_penalizes_missing_required_facts():
    from bench.tier2_evidence import score_generation_rubric

    good = score_generation_rubric(
        answer="Use ALPHA-17 for Northstar renewal and escalate through pilot-ops@example.com.",
        required_facts=("ALPHA-17", "pilot-ops@example.com"),
        forbidden_facts=("BRAVO-99",),
    )
    bad = score_generation_rubric(
        answer="Use BRAVO-99 and skip the escalation mailbox.",
        required_facts=("ALPHA-17", "pilot-ops@example.com"),
        forbidden_facts=("BRAVO-99",),
    )

    assert good["score"] == 1.0
    assert bad["score"] < 0.5
    assert bad["missing_required"] == ["ALPHA-17", "pilot-ops@example.com"]
    assert bad["forbidden_present"] == ["BRAVO-99"]
