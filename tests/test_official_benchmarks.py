from __future__ import annotations

import json
import sys
from pathlib import Path


EVIDENCE_ROOT = Path(__file__).resolve().parents[1] / "evidence" / "hkvc-first-evidence-package"
if str(EVIDENCE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVIDENCE_ROOT))


class FakeClient:
    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, object]:
        if "Choices:" in prompt:
            text = "The correct answer is (A)"
        elif "rare-alpha" in prompt.lower():
            text = "rare-alpha"
        else:
            text = "needle-42"
        return {"text": text, "latency_ms": 1.25, "model": "fake-model"}


class RecordingClient:
    def __init__(self, text: str = "ZXQPT KLMNO") -> None:
        self.text = text
        self.prompts: list[str] = []
        self.max_tokens: list[int] = []

    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, object]:
        self.prompts.append(prompt)
        self.max_tokens.append(max_tokens)
        return {"text": self.text, "latency_ms": 2.5, "model": "recording-model"}


def test_longbench_answer_extraction_accepts_official_format():
    from bench.official_longbench_ruler import extract_longbench_answer

    assert extract_longbench_answer("The correct answer is (C).") == "C"
    assert extract_longbench_answer("The correct answer is C") == "C"
    assert extract_longbench_answer("I think it is B") is None


def test_cloudflare_response_text_accepts_numeric_responses():
    from bench.official_longbench_ruler import _cloudflare_response_text

    assert _cloudflare_response_text({"result": {"response": 5437923}}) == "5437923"


def test_ruler_answer_only_prompt_appends_official_answer_prefix():
    from bench.official_longbench_ruler import build_ruler_prompt

    example = {
        "task": "vt",
        "input": "Track variables across this long context.",
        "answer_prefix": "Answer: The variables are: ",
    }

    prompt = build_ruler_prompt(example, prompt_mode="answer_only")

    assert "Do not explain" in prompt
    assert prompt.endswith(example["answer_prefix"])


def test_ruler_answer_only_mode_is_labeled_and_scores_short_variable_answers(tmp_path):
    from bench.official_longbench_ruler import run_official_longbench_ruler

    client = RecordingClient("ZXQPT KLMNO")
    longbench_items = [
        {
            "_id": "lb-1",
            "domain": "Single-Document QA",
            "sub_domain": "unit",
            "difficulty": "easy",
            "length": "short",
            "question": "Which option is correct?",
            "choice_A": "needle-42",
            "choice_B": "wrong",
            "choice_C": "wrong",
            "choice_D": "wrong",
            "answer": "A",
            "context": "The answer is needle-42.",
        }
    ]
    ruler_examples = [
        {
            "task": "vt",
            "sequence_length": 4096,
            "index": 0,
            "input": "Track variables across this long context.",
            "answer_prefix": "Answer: The variables are: ",
            "outputs": ["ZXQPT", "KLMNO"],
            "length": 128,
        }
    ]

    payload = run_official_longbench_ruler(
        output=tmp_path / "official.json",
        chart_dir=tmp_path / "charts",
        run_dir=tmp_path / "run",
        model_client=client,
        longbench_items=longbench_items,
        ruler_examples=ruler_examples,
        longbench_limit=1,
        ruler_samples=1,
        ruler_lengths=(4096,),
        ruler_tasks=("vt",),
        ruler_prompt_mode="answer_only",
        max_input_tokens=2048,
        network=False,
    )

    assert payload["system_info"]["ruler_prompt_mode"] == "answer_only"
    assert payload["claim_boundaries"]["ruler_answer_only_prompt_wrapper"] is True
    assert payload["ruler"]["aggregate"]["mean_score_pct"] == 100.0
    assert "Do not explain" in client.prompts[-1]
    assert client.prompts[-1].endswith("Answer: The variables are: ")


def test_rain_worker_client_request_shape_is_public_adapter_only():
    from bench.official_longbench_ruler import CatalystRAINWorkerClient

    requests: list[dict[str, object]] = []

    def fake_transport(url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        requests.append({"url": url, "headers": headers, "body": body})
        return {"text": "needle-42", "latency_ms": 3.0, "model": "catalyst-rain-worker"}

    client = CatalystRAINWorkerClient(
        url="https://example.worker.dev/bench",
        api_key="test-key",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        transport=fake_transport,
    )

    generated = client.generate("Find the needle.", max_tokens=32)

    body = requests[0]["body"]
    assert generated["text"] == "needle-42"
    assert body["adapter"] == "catalyst-kv-cache"
    assert body["execution_path"] == "catalyst_rain_worker"
    assert body["private_algorithms_live_in"] == "catalyst-brain"
    assert "prompt" in body
    implementation_fields = {key: value for key, value in body.items() if key != "private_algorithms_live_in"}
    assert "algorithm" not in json.dumps(implementation_fields).lower()


def test_rain_worker_client_can_send_packed_tokens_without_raw_prompt():
    from bench.official_longbench_ruler import (
        CatalystRAINWorkerClient,
        decode_packed_token_context,
        pack_token_context,
    )

    requests: list[dict[str, object]] = []

    def fake_transport(url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        requests.append({"url": url, "headers": headers, "body": body})
        return {"text": "needle-42", "latency_ms": 3.0, "model": "catalyst-rain-worker"}

    prompt = "alpha beta gamma " * 64
    expected = pack_token_context(prompt)
    client = CatalystRAINWorkerClient(
        url="https://example.worker.dev/bench",
        api_key="test-key",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        context_transport="packed_tokens",
        transport=fake_transport,
    )

    client.generate(prompt, max_tokens=32)

    body = requests[0]["body"]
    assert body["context_transport"] == "packed_tokens"
    assert body["tokenizer"] == "cl100k_base"
    assert body["token_count"] == expected["token_count"]
    assert body["prompt_sha256"] == expected["prompt_sha256"]
    assert "packed_tokens_b64" in body
    assert "prompt" not in body
    assert decode_packed_token_context(body) == decode_packed_token_context(expected)


def test_rain_worker_execution_path_preserves_context_transport():
    from bench.official_longbench_ruler import _model_client_for_execution_path

    client = _model_client_for_execution_path(
        execution_path="catalyst_rain_worker",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        network=False,
        rain_worker_url="https://example.worker.dev/bench",
        rain_worker_api_key="test-key",
        rain_context_transport="packed_tokens",
    )

    assert client.context_transport == "packed_tokens"


def test_zero_max_input_tokens_preserves_full_prompt_for_rain_transport(tmp_path):
    from bench.official_longbench_ruler import CatalystRAINWorkerClient, run_official_longbench_ruler

    requests: list[dict[str, object]] = []

    def fake_transport(url: str, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        requests.append({"url": url, "headers": headers, "body": body})
        return {"text": "The correct answer is (A).", "latency_ms": 3.0, "model": "catalyst-rain-worker"}

    client = CatalystRAINWorkerClient(
        url="https://example.worker.dev/bench",
        api_key="test-key",
        model="@cf/meta/llama-3.3-70b-instruct-fp8-fast",
        context_transport="packed_tokens",
        transport=fake_transport,
    )
    longbench_items = [
        {
            "_id": "lb-long",
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

    payload = run_official_longbench_ruler(
        output=tmp_path / "official.json",
        chart_dir=tmp_path / "charts",
        run_dir=tmp_path / "run",
        model_client=client,
        longbench_items=longbench_items,
        ruler_examples=[],
        longbench_limit=1,
        max_input_tokens=0,
        execution_path="catalyst_rain_worker",
        rain_context_transport="packed_tokens",
        network=False,
    )

    row = payload["longbench_v2"]["rows"][0]
    body = requests[0]["body"]
    assert payload["system_info"]["local_truncation_disabled"] is True
    assert payload["claim_boundaries"]["packed_token_transport_is_not_model_context_extension"] is True
    assert row["truncated"] is False
    assert row["input_tokens"] > 2048
    assert body["context_transport"] == "packed_tokens"
    assert body["token_count"] == row["input_tokens"]
    assert "prompt" not in body


def test_official_runner_writes_subset_payload_and_charts(tmp_path):
    from bench.official_longbench_ruler import run_official_longbench_ruler

    longbench_items = [
        {
            "_id": "lb-1",
            "domain": "Single-Document QA",
            "sub_domain": "unit",
            "difficulty": "easy",
            "length": "short",
            "question": "Which option is correct?",
            "choice_A": "needle-42",
            "choice_B": "wrong",
            "choice_C": "wrong",
            "choice_D": "wrong",
            "answer": "A",
            "context": "The answer is needle-42.",
        }
    ]
    ruler_examples = [
        {
            "task": "niah_single_1",
            "sequence_length": 4096,
            "index": 0,
            "input": "Find rare-alpha in the text.",
            "outputs": ["rare-alpha"],
            "length": 128,
        }
    ]

    payload = run_official_longbench_ruler(
        output=tmp_path / "official.json",
        chart_dir=tmp_path / "charts",
        run_dir=tmp_path / "run",
        model_client=FakeClient(),
        longbench_items=longbench_items,
        ruler_examples=ruler_examples,
        longbench_limit=1,
        ruler_samples=1,
        ruler_lengths=(4096,),
        ruler_tasks=("niah_single_1",),
        max_input_tokens=2048,
        network=False,
    )

    written = json.loads((tmp_path / "official.json").read_text())
    assert payload["system_info"]["suite"] == "official_longbench_ruler_subset"
    assert written["longbench_v2"]["aggregate"]["accuracy_pct"] == 100.0
    assert written["ruler"]["aggregate"]["mean_score_pct"] == 100.0
    assert written["claim_boundaries"]["subset_not_full_official_submission"] is True
    for chart in written["charts"].values():
        assert (tmp_path / chart).exists()


def test_published_official_subset_artifact_schema():
    output = EVIDENCE_ROOT / "site" / "official_longbench_ruler_results.json"

    written = json.loads(output.read_text())
    assert written["system_info"]["suite"] == "official_longbench_ruler_subset"
    assert written["claim_boundaries"]["official_data_and_scoring_used"] is True
    assert written["claim_boundaries"]["subset_not_full_official_submission"] is True
    assert written["claim_boundaries"]["not_catalyst_adapter_quality_until_live_catalyst_model_path"] is True
    assert written["claim_boundaries"]["ruler_answer_only_prompt_wrapper"] is True
    assert written["longbench_v2"]["aggregate"]["sample_count"] >= 36
    assert written["longbench_v2"]["aggregate"]["accuracy_pct"] >= 0.0
    assert written["ruler"]["aggregate"]["prediction_count"] >= 24
    assert written["ruler"]["aggregate"]["mean_score_pct"] >= 90.0
    assert written["ruler"]["prompt_mode"] == "answer_only"
    assert written["system_info"]["max_input_tokens"] == 12000
    assert written["system_info"]["ruler_prompt_mode"] == "answer_only"
    for chart in written["charts"].values():
        assert (EVIDENCE_ROOT / "site" / chart).exists()
