from __future__ import annotations

import json


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
