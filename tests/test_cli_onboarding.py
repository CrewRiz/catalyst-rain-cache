from __future__ import annotations

import json
import subprocess
import sys


def test_doctor_json_reports_sdk_boundary(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["doctor", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter_package"] == "catalyst-kv-cache"
    assert payload["public_boundary"]["algorithms_live_in"] == "catalyst-brain"
    assert payload["public_boundary"]["ships_private_algorithms"] is False
    assert "run_hkvc_tier2_evidence" in payload["sdk_features"]
    assert "run_hkvc_lossless_equivalence_benchmark" in payload["sdk_features"]
    assert "run_hkvc_lossless_scale_benchmark" in payload["sdk_features"]


def test_onboard_json_gives_dropin_next_steps(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["onboard", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["goal"] == "drop_in_long_context_private_ai"
    assert "create_transformers_cache" in payload["python_api"]
    assert "catalyst-kv-cache doctor" in payload["commands"]
    assert any("bench.lossless_scale" in command for command in payload["commands"])


def test_demo_json_summarizes_publishable_evidence(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["demo", "--json"]) == 0

    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert captured.err == ""
    assert payload["demo_ready"] is True
    assert payload["public_boundary"]["ships_private_algorithms"] is False
    assert payload["official_subset"]["longbench_v2"]["sample_count"] >= 36
    assert payload["official_subset"]["ruler"]["prediction_count"] >= 24
    assert payload["live_model_probe"]["status"] == "measured"
    assert "catalyst-kv-cache demo --json" in payload["commands"]


def test_demo_json_surfaces_claim_guardrails(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["demo", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    guardrails = payload["do_not_claim_yet"]

    assert len(guardrails) >= 7
    assert any("compact semantic state" in claim and "full KV tensors" in claim for claim in guardrails)
    assert any("without exact-state transport" in claim for claim in guardrails)
    assert any("Official LongBench/RULER production scores" in claim for claim in guardrails)
    assert any("Chunked archives" in claim and "hot O(1) decode" in claim for claim in guardrails)
    assert any("operator profiles" in claim and "full model generation-quality" in claim for claim in guardrails)
    assert any("Cloudflare Workers AI smoke probe" in claim for claim in guardrails)
    assert any("Python prototype latency" in claim and "production kernel throughput" in claim for claim in guardrails)


def test_demo_human_output_mentions_guardrails(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["demo"]) == 0

    output = capsys.readouterr().out
    assert "Claim guardrails:" in output
    assert "demo --json" in output


def test_demo_payload_does_not_import_sdk(monkeypatch):
    import catalyst_kv_cache.sdk_bridge as sdk_bridge

    def fail_load():
        raise AssertionError("demo should not import catalyst-brain")

    monkeypatch.setattr(sdk_bridge, "load_catalyst_brain", fail_load)

    payload = sdk_bridge.demo_payload()

    assert payload["demo_ready"] is True
    assert payload["public_boundary"]["algorithms_live_in"] == "catalyst-brain"


def test_package_import_is_lazy_for_demo_surface():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, catalyst_kv_cache; print('catalyst_brain' in sys.modules)",
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"
    assert result.stderr == ""


def test_legacy_cli_without_subcommand_runs_smoke(capsys):
    from catalyst_kv_cache.cli import main

    assert main(["--mode", "refs"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["mode"] == "refs"
    assert payload["seq_length"] > 0


def test_main_uses_sys_argv_when_called_as_console_script(monkeypatch, capsys):
    from catalyst_kv_cache.cli import main

    monkeypatch.setattr("sys.argv", ["catalyst-kv-cache", "doctor", "--json"])

    assert main() == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["adapter_package"] == "catalyst-kv-cache"
