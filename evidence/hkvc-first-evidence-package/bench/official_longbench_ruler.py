from __future__ import annotations

import argparse
import base64
import gzip
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import urllib.request
from array import array
from pathlib import Path
from typing import Any, Iterable, Sequence


DEFAULT_INDUSTRY_ROOT = Path(os.environ.get("CATALYST_BENCHMARK_ROOT", "/Users/ghostmesh/benchmark-runs/industry"))
DEFAULT_LONGBENCH_REPO = DEFAULT_INDUSTRY_ROOT / "LongBench"
DEFAULT_RULER_REPO = DEFAULT_INDUSTRY_ROOT / "RULER"
DEFAULT_MODEL = "@cf/meta/llama-3.3-70b-instruct-fp8-fast"
DEFAULT_LONGBENCH_LIMIT = 36
DEFAULT_MAX_INPUT_TOKENS = 64_000
DEFAULT_EXECUTION_PATH = "cloudflare_workers_ai"
DEFAULT_RULER_PROMPT_MODE = "direct"
DEFAULT_RULER_TASKS = (
    "niah_single_1",
    "niah_multikey_2",
    "niah_multikey_3",
    "vt",
    "cwe",
    "fwe",
)
DEFAULT_RULER_LENGTHS = (4096, 8192)


def run_official_longbench_ruler(
    *,
    output: str | Path,
    chart_dir: str | Path,
    run_dir: str | Path,
    longbench_repo: str | Path = DEFAULT_LONGBENCH_REPO,
    ruler_repo: str | Path = DEFAULT_RULER_REPO,
    model: str = DEFAULT_MODEL,
    longbench_limit: int = DEFAULT_LONGBENCH_LIMIT,
    max_input_tokens: int = DEFAULT_MAX_INPUT_TOKENS,
    ruler_tasks: Sequence[str] = DEFAULT_RULER_TASKS,
    ruler_lengths: Sequence[int] = DEFAULT_RULER_LENGTHS,
    ruler_samples: int = 2,
    ruler_prompt_mode: str = DEFAULT_RULER_PROMPT_MODE,
    execution_path: str = DEFAULT_EXECUTION_PATH,
    rain_worker_url: str | None = None,
    rain_worker_api_key: str | None = None,
    rain_context_transport: str = "plain_prompt",
    network: bool = True,
    model_client: Any | None = None,
    longbench_items: Sequence[dict[str, Any]] | None = None,
    ruler_examples: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_path = Path(output)
    chart_path = Path(chart_dir)
    run_path = Path(run_dir)
    longbench_path = Path(longbench_repo)
    ruler_path = Path(ruler_repo)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chart_path.mkdir(parents=True, exist_ok=True)
    run_path.mkdir(parents=True, exist_ok=True)

    client = model_client or _model_client_for_execution_path(
        execution_path=execution_path,
        model=model,
        network=network,
        rain_worker_url=rain_worker_url,
        rain_worker_api_key=rain_worker_api_key,
        rain_context_transport=rain_context_transport,
    )
    lb_items = list(longbench_items) if longbench_items is not None else _load_longbench_items(
        longbench_repo=longbench_path,
        limit=longbench_limit,
    )
    ruler_rows = list(ruler_examples) if ruler_examples is not None else _prepare_ruler_examples(
        ruler_repo=ruler_path,
        run_dir=run_path / "ruler",
        tasks=tuple(ruler_tasks),
        lengths=tuple(int(length) for length in ruler_lengths),
        samples=ruler_samples,
    )

    longbench_result = _run_longbench_v2(
        items=lb_items,
        longbench_repo=longbench_path,
        client=client,
        max_input_tokens=max_input_tokens,
    )
    ruler_result = _run_ruler(
        examples=ruler_rows,
        client=client,
        prompt_mode=ruler_prompt_mode,
    )
    remaining_blockers = [
        "Run all 503 LongBench v2 examples before claiming a full official LongBench score.",
        "Run the full RULER task matrix and target context lengths before claiming a full official RULER score.",
        "Connect the official runners to a live Catalyst/catalyst-brain model path before claiming Catalyst adapter quality.",
    ]
    if ruler_prompt_mode != "direct":
        remaining_blockers.append(
            "For leaderboard-style RULER comparisons, rerun with the benchmark's unmodified prompt template."
        )
    if max_input_tokens <= 12_000:
        remaining_blockers.insert(
            2,
            "Workers AI direct REST returned 413 payload errors at 32K/64K LongBench prompt caps in this environment; this artifact uses a 12K cap.",
        )
    payload = {
        "system_info": {
            "suite": "official_longbench_ruler_subset",
            "python": platform.python_version(),
            "platform": platform.platform(),
            "model": model,
            "longbench_limit": longbench_limit,
            "max_input_tokens": max_input_tokens,
            "local_truncation_disabled": max_input_tokens <= 0,
            "execution_path": execution_path,
            "rain_context_transport": rain_context_transport if execution_path == "catalyst_rain_worker" else None,
            "ruler_tasks": list(ruler_tasks),
            "ruler_lengths": [int(length) for length in ruler_lengths],
            "ruler_samples": ruler_samples,
            "ruler_prompt_mode": ruler_prompt_mode,
            "run_dir": str(run_path),
            "longbench_repo": str(longbench_path),
            "ruler_repo": str(ruler_path),
            "longbench_revision": _git_revision(longbench_path),
            "ruler_revision": _git_revision(ruler_path),
        },
        "claim_boundaries": {
            "subset_not_full_official_submission": True,
            "official_data_and_scoring_used": True,
            "cloudflare_workers_ai_model_path": True,
            "catalyst_rain_worker_path": execution_path == "catalyst_rain_worker",
            "ruler_answer_only_prompt_wrapper": ruler_prompt_mode == "answer_only",
            "packed_token_transport_is_not_model_context_extension": rain_context_transport == "packed_tokens",
            "not_catalyst_adapter_quality_until_live_catalyst_model_path": True,
            "private_algorithms_remain_in": "catalyst_brain",
        },
        "longbench_v2": longbench_result,
        "ruler": ruler_result,
        "remaining_blockers": remaining_blockers,
        "charts": {
            "official_longbench_accuracy": "charts/official_longbench_accuracy.svg",
            "official_ruler_accuracy": "charts/official_ruler_accuracy.svg",
        },
    }
    _write_charts(payload, chart_path)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


class CloudflareWorkersAIClient:
    def __init__(self, *, model: str, network: bool = True) -> None:
        self.model = model
        self.network = network
        self.account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
        self.api_token = os.environ.get("CLOUDFLARE_API_TOKEN") or os.environ.get("CLOUDFLARE_AUTH_TOKEN")

    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        if not self.network:
            return {"text": "", "latency_ms": None, "model": self.model, "status": "skipped_network_disabled"}
        if not (self.account_id and self.api_token):
            raise RuntimeError("CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required for official benchmark inference.")
        url = f"https://api.cloudflare.com/client/v4/accounts/{self.account_id}/ai/run/{self.model}"
        body = {
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        request = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        started = time.perf_counter_ns()
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=180) as response:
                    decoded = json.loads(response.read().decode("utf-8"))
                return {
                    "text": _cloudflare_response_text(decoded),
                    "latency_ms": round((time.perf_counter_ns() - started) / 1_000_000.0, 4),
                    "model": self.model,
                    "status": "ok",
                    "attempts": attempt + 1,
                }
            except Exception as exc:  # pragma: no cover - network dependent
                last_error = exc
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"Cloudflare Workers AI request failed after retries: {last_error!r}")


class CatalystRAINWorkerClient:
    """Public benchmark client for a private Catalyst Worker/.rain serving path."""

    def __init__(
        self,
        *,
        url: str,
        api_key: str | None,
        model: str,
        transport: Any | None = None,
        rain_payload: str | None = None,
        context_transport: str = "plain_prompt",
    ) -> None:
        self.url = url
        self.api_key = api_key
        self.model = model
        self.transport = transport or _post_json
        self.rain_payload = rain_payload
        self.context_transport = "rain" if rain_payload else context_transport

    def generate(self, prompt: str, *, max_tokens: int) -> dict[str, Any]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        body = {
            "adapter": "catalyst-kv-cache",
            "execution_path": "catalyst_rain_worker",
            "private_algorithms_live_in": "catalyst-brain",
            "context_transport": self.context_transport,
            "model": self.model,
            "max_tokens": max_tokens,
        }
        if self.rain_payload:
            body["rain_payload"] = self.rain_payload
        elif self.context_transport == "packed_tokens":
            body.update(pack_token_context(prompt))
        elif self.context_transport == "plain_prompt":
            body["prompt"] = prompt
        else:
            raise ValueError(f"Unknown Catalyst Worker context transport: {self.context_transport}")
        response = self.transport(self.url, headers, body)
        if not isinstance(response, dict):
            raise RuntimeError("Catalyst RAIN Worker transport must return a JSON object.")
        return response


def pack_token_context(prompt: str, *, tokenizer: str = "cl100k_base") -> dict[str, Any]:
    import tiktoken

    encoding = tiktoken.get_encoding(tokenizer)
    token_ids = encoding.encode(prompt, disallowed_special=())
    packed = array("I", token_ids)
    if sys.byteorder != "little":
        packed.byteswap()
    raw = packed.tobytes()
    compressed = gzip.compress(raw)
    return {
        "tokenizer": tokenizer,
        "token_codec": "uint32le+gzip+base64",
        "token_count": len(token_ids),
        "packed_tokens_b64": base64.b64encode(compressed).decode("ascii"),
        "prompt_sha256": _sha256(prompt),
    }


def decode_packed_token_context(payload: dict[str, Any]) -> list[int]:
    if payload.get("token_codec") != "uint32le+gzip+base64":
        raise ValueError(f"Unsupported token codec: {payload.get('token_codec')}")
    compressed = base64.b64decode(str(payload["packed_tokens_b64"]))
    raw = gzip.decompress(compressed)
    token_ids = array("I")
    token_ids.frombytes(raw)
    if sys.byteorder != "little":
        token_ids.byteswap()
    return list(token_ids)


def _model_client_for_execution_path(
    *,
    execution_path: str,
    model: str,
    network: bool,
    rain_worker_url: str | None,
    rain_worker_api_key: str | None,
    rain_context_transport: str = "plain_prompt",
) -> Any:
    if execution_path == "cloudflare_workers_ai":
        return CloudflareWorkersAIClient(model=model, network=network)
    if execution_path == "catalyst_rain_worker":
        url = rain_worker_url or os.environ.get("CATALYST_RAIN_WORKER_URL")
        if not url:
            raise RuntimeError("CATALYST_RAIN_WORKER_URL is required for the Catalyst RAIN Worker execution path.")
        return CatalystRAINWorkerClient(
            url=url,
            api_key=rain_worker_api_key or os.environ.get("CATALYST_RAIN_WORKER_API_KEY"),
            model=model,
            context_transport=rain_context_transport,
        )
    raise ValueError(f"Unknown execution path: {execution_path}")


def _post_json(url: str, headers: dict[str, str], body: dict[str, Any]) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=300) as response:  # pragma: no cover - network dependent
        decoded = json.loads(response.read().decode("utf-8"))
    decoded.setdefault("latency_ms", round((time.perf_counter_ns() - started) / 1_000_000.0, 4))
    decoded.setdefault("model", body.get("model"))
    return decoded


def extract_longbench_answer(response: str) -> str | None:
    cleaned = response.replace("*", "")
    match = re.search(r"The correct answer is \(([A-D])\)", cleaned)
    if match:
        return match.group(1)
    match = re.search(r"The correct answer is ([A-D])", cleaned)
    return match.group(1) if match else None


def _run_longbench_v2(
    *,
    items: Sequence[dict[str, Any]],
    longbench_repo: Path,
    client: Any,
    max_input_tokens: int,
) -> dict[str, Any]:
    template = (longbench_repo / "prompts" / "0shot.txt").read_text(encoding="utf-8")
    rows = []
    for item in items:
        prompt = _longbench_prompt(template, item)
        prompt, input_tokens, truncated = _truncate_prompt(prompt, max_input_tokens=max_input_tokens)
        generated = client.generate(prompt, max_tokens=128)
        response = str(generated.get("text", "")).strip()
        pred = extract_longbench_answer(response)
        rows.append(
            {
                "id": item["_id"],
                "domain": item["domain"],
                "sub_domain": item["sub_domain"],
                "difficulty": item["difficulty"],
                "length": item["length"],
                "answer": item["answer"],
                "prediction": pred,
                "judge": pred == item["answer"],
                "response": response[:500],
                "input_tokens": input_tokens,
                "truncated": truncated,
                "latency_ms": generated.get("latency_ms"),
                "prompt_sha256": _sha256(prompt),
            }
        )
    return {
        "source": "THUDM/LongBench-v2",
        "source_url": "https://github.com/THUDM/LongBench",
        "dataset_id": "THUDM/LongBench-v2",
        "mode": "official_v2_subset",
        "aggregate": _longbench_aggregate(rows),
        "rows": rows,
    }


def build_ruler_prompt(example: dict[str, Any], *, prompt_mode: str = DEFAULT_RULER_PROMPT_MODE) -> str:
    prompt = str(example["input"])
    if prompt_mode == "direct":
        return prompt
    if prompt_mode == "answer_only":
        answer_prefix = str(example.get("answer_prefix") or "")
        instruction = "\n\nDo not explain. Return only the requested answer values separated by spaces."
        return f"{prompt}{instruction}{answer_prefix}"
    raise ValueError(f"Unknown RULER prompt mode: {prompt_mode}")


def _run_ruler(*, examples: Sequence[dict[str, Any]], client: Any, prompt_mode: str) -> dict[str, Any]:
    predicted_rows = []
    for example in examples:
        prompt = build_ruler_prompt(example, prompt_mode=prompt_mode)
        generated = client.generate(prompt, max_tokens=_ruler_max_tokens(example["task"]))
        pred = str(generated.get("text", "")).strip()
        score = _ruler_score([pred], [example["outputs"]], task=example["task"])
        predicted_rows.append(
            {
                "task": example["task"],
                "sequence_length": int(example["sequence_length"]),
                "index": example["index"],
                "outputs": example["outputs"],
                "prediction": pred[:500],
                "score_pct": score,
                "latency_ms": generated.get("latency_ms"),
                "input_tokens": example.get("length"),
                "prompt_mode": prompt_mode,
                "prompt_sha256": _sha256(prompt),
            }
        )
    grouped = {}
    for row in predicted_rows:
        key = (row["task"], row["sequence_length"])
        grouped.setdefault(key, []).append(row)
    task_rows = []
    for (task, length), rows in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        task_rows.append(
            {
                "task": task,
                "sequence_length": length,
                "sample_count": len(rows),
                "score_pct": round(statistics.fmean(row["score_pct"] for row in rows), 2),
                "null_predictions": sum(1 for row in rows if not row["prediction"]),
                "mean_latency_ms": _safe_mean(row["latency_ms"] for row in rows),
                "mean_input_tokens": _safe_mean(row["input_tokens"] for row in rows),
            }
        )
    return {
        "source": "NVIDIA/RULER",
        "source_url": "https://github.com/NVIDIA/RULER",
        "benchmark": "synthetic",
        "mode": "official_generated_subset",
        "prompt_mode": prompt_mode,
        "aggregate": {
            "task_length_count": len(task_rows),
            "prediction_count": len(predicted_rows),
            "mean_score_pct": round(statistics.fmean(row["score_pct"] for row in task_rows), 2) if task_rows else 0.0,
            "min_score_pct": min((row["score_pct"] for row in task_rows), default=0.0),
            "max_score_pct": max((row["score_pct"] for row in task_rows), default=0.0),
        },
        "task_rows": task_rows,
        "prediction_rows": predicted_rows,
    }


def _load_longbench_items(*, longbench_repo: Path, limit: int) -> list[dict[str, Any]]:
    from datasets import load_dataset

    dataset = load_dataset("THUDM/LongBench-v2", split="train")
    rows = [dict(item) for item in dataset]
    selected: list[dict[str, Any]] = []
    seen = set()
    for group_key in _longbench_group_order(rows):
        for row in rows:
            key = (row["domain"], row["length"], row["difficulty"])
            if key == group_key and row["_id"] not in seen:
                selected.append(row)
                seen.add(row["_id"])
                break
            if len(selected) >= limit:
                break
        if len(selected) >= limit:
            break
    if len(selected) < limit:
        for row in rows:
            if row["_id"] not in seen:
                selected.append(row)
                seen.add(row["_id"])
            if len(selected) >= limit:
                break
    return selected[:limit]


def _prepare_ruler_examples(
    *,
    ruler_repo: Path,
    run_dir: Path,
    tasks: Sequence[str],
    lengths: Sequence[int],
    samples: int,
) -> list[dict[str, Any]]:
    data_dir = run_dir / "data"
    examples: list[dict[str, Any]] = []
    for length in lengths:
        length_dir = data_dir / str(length)
        for task in tasks:
            task_dir = length_dir / task
            if task_dir.exists():
                for path in task_dir.glob("*.jsonl"):
                    path.unlink()
            _run_ruler_prepare(
                ruler_repo=ruler_repo,
                save_dir=length_dir,
                task=task,
                max_seq_length=int(length),
                samples=samples,
            )
            data_path = length_dir / task / "validation.jsonl"
            for row in _read_jsonl(data_path):
                examples.append(
                    {
                        "task": task,
                        "sequence_length": int(length),
                        "index": row["index"],
                        "input": row["input"],
                        "outputs": row["outputs"],
                        "length": row.get("length"),
                        "answer_prefix": row.get("answer_prefix", ""),
                    }
                )
    return examples


def _run_ruler_prepare(*, ruler_repo: Path, save_dir: Path, task: str, max_seq_length: int, samples: int) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ruler_repo / 'scripts' / 'data'}{os.pathsep}{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        str(ruler_repo / "scripts" / "data" / "prepare.py"),
        "--save_dir",
        str(save_dir),
        "--benchmark",
        "synthetic",
        "--task",
        task,
        "--tokenizer_path",
        "cl100k_base",
        "--tokenizer_type",
        "openai",
        "--max_seq_length",
        str(max_seq_length),
        "--model_template_type",
        "base",
        "--num_samples",
        str(samples),
    ]
    result = subprocess.run(
        command,
        cwd=str(ruler_repo / "scripts"),
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    data_path = save_dir / task / "validation.jsonl"
    if result.returncode != 0 or not data_path.exists():
        raise RuntimeError(f"RULER prepare failed for {task}@{max_seq_length}: {result.stderr or result.stdout}")


def _longbench_group_order(rows: Sequence[dict[str, Any]]) -> list[tuple[str, str, str]]:
    groups = sorted({(row["domain"], row["length"], row["difficulty"]) for row in rows})
    length_rank = {"short": 0, "medium": 1, "long": 2}
    difficulty_rank = {"easy": 0, "hard": 1}
    return sorted(groups, key=lambda item: (length_rank.get(item[1], 99), difficulty_rank.get(item[2], 99), item[0]))


def _longbench_prompt(template: str, item: dict[str, Any]) -> str:
    return (
        template.replace("$DOC$", item["context"].strip())
        .replace("$Q$", item["question"].strip())
        .replace("$C_A$", item["choice_A"].strip())
        .replace("$C_B$", item["choice_B"].strip())
        .replace("$C_C$", item["choice_C"].strip())
        .replace("$C_D$", item["choice_D"].strip())
    )


def _truncate_prompt(prompt: str, *, max_input_tokens: int) -> tuple[str, int, bool]:
    import tiktoken

    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(prompt, disallowed_special=())
    if max_input_tokens <= 0:
        return prompt, len(tokens), False
    if len(tokens) <= max_input_tokens:
        return prompt, len(tokens), False
    half = max_input_tokens // 2
    kept = tokens[:half] + tokens[-half:]
    return encoding.decode(kept), len(kept), True


def _longbench_aggregate(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    by_difficulty = _accuracy_by_key(rows, "difficulty")
    by_length = _accuracy_by_key(rows, "length")
    by_domain = _accuracy_by_key(rows, "domain")
    return {
        "sample_count": count,
        "accuracy_pct": _accuracy(rows),
        "easy_accuracy_pct": by_difficulty.get("easy"),
        "hard_accuracy_pct": by_difficulty.get("hard"),
        "short_accuracy_pct": by_length.get("short"),
        "medium_accuracy_pct": by_length.get("medium"),
        "long_accuracy_pct": by_length.get("long"),
        "by_domain_accuracy_pct": by_domain,
        "truncated_count": sum(1 for row in rows if row["truncated"]),
        "mean_latency_ms": _safe_mean(row["latency_ms"] for row in rows),
        "mean_input_tokens": _safe_mean(row["input_tokens"] for row in rows),
    }


def _accuracy(rows: Sequence[dict[str, Any]]) -> float:
    return round(100.0 * sum(1 for row in rows if row["judge"]) / max(1, len(rows)), 2)


def _accuracy_by_key(rows: Sequence[dict[str, Any]], key: str) -> dict[str, float]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return {name: _accuracy(items) for name, items in sorted(groups.items())}


def _ruler_score(preds: Sequence[str], refs: Sequence[Sequence[str]], *, task: str) -> float:
    if task.startswith("qa"):
        score = sum(max(1.0 if ref.lower() in pred.lower() else 0.0 for ref in row_refs) for pred, row_refs in zip(preds, refs))
        return round(score / max(1, len(preds)) * 100.0, 2)
    score = 0.0
    for pred, row_refs in zip(preds, refs):
        score += sum(1.0 if ref.lower() in pred.lower() else 0.0 for ref in row_refs) / max(1, len(row_refs))
    return round(score / max(1, len(preds)) * 100.0, 2)


def _ruler_max_tokens(task: str) -> int:
    if task == "cwe":
        return 120
    if task == "fwe":
        return 50
    if task == "vt":
        return 30
    return 128


def _safe_mean(values: Iterable[Any]) -> float | None:
    numeric = [float(value) for value in values if isinstance(value, (int, float))]
    return round(statistics.fmean(numeric), 4) if numeric else None


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _cloudflare_response_text(decoded: dict[str, Any]) -> str:
    result = decoded.get("result", decoded)
    if isinstance(result, str):
        return result
    if isinstance(result, dict):
        for key in ("response", "text", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (int, float)):
                return str(value)
        choices = result.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                message = first.get("message")
                if isinstance(message, dict) and isinstance(message.get("content"), str):
                    return message["content"]
                if isinstance(first.get("text"), str):
                    return first["text"]
    return json.dumps(decoded, sort_keys=True)[:500]


def _write_charts(payload: dict[str, Any], chart_dir: Path) -> None:
    lb = payload["longbench_v2"]["aggregate"]
    ruler = payload["ruler"]
    (chart_dir / "official_longbench_accuracy.svg").write_text(
        _bar_chart(
            "Official LongBench v2 Subset",
            [
                ("Overall", lb["accuracy_pct"], "#0f766e"),
                ("Easy", lb["easy_accuracy_pct"] or 0.0, "#2563eb"),
                ("Hard", lb["hard_accuracy_pct"] or 0.0, "#7c2d12"),
            ],
            max_value=100.0,
        )
    )
    values = [(f'{row["task"]}@{row["sequence_length"] // 1000}K', row["score_pct"], "#4f46e5") for row in ruler["task_rows"]]
    (chart_dir / "official_ruler_accuracy.svg").write_text(
        _bar_chart("Official RULER Generated Subset", values[:12], max_value=100.0)
    )


def _bar_chart(title: str, values: Sequence[tuple[str, float, str]], *, max_value: float = 100.0) -> str:
    width = 760
    height = 430
    if not values:
        values = [("No rows", 0.0, "#64748b")]
    step = 620 / max(1, len(values))
    bar_width = min(68, max(22, step * 0.56))
    body = []
    for index, (label, value, color) in enumerate(values):
        x = 70 + index * step + (step - bar_width) / 2
        bar_height = max(3.0, 235.0 * float(value) / max(max_value, 1e-12))
        body.append(f'<rect x="{x:.1f}" y="{330 - bar_height:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="5" fill="{color}"/>')
        body.append(f'<text x="{x + bar_width / 2:.1f}" y="{max(20, 320 - bar_height):.1f}" text-anchor="middle" font-size="10" fill="#111827">{float(value):.1f}</text>')
        body.append(f'<text x="{x + bar_width / 2:.1f}" y="364" text-anchor="middle" font-size="9" fill="#111827">{label}</text>')
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        '<rect width="100%" height="100%" fill="#fbfaf7"/>'
        f'<text x="56" y="44" font-family="Inter, Helvetica, Arial, sans-serif" font-size="24" font-weight="700" fill="#111827">{title}</text>'
        '<text x="56" y="70" font-family="Inter, Helvetica, Arial, sans-serif" font-size="13" fill="#4b5563">Generated from official benchmark data/scoring with Workers AI inference.</text>'
        '<line x1="70" y1="330" x2="690" y2="330" stroke="#d1d5db"/>'
        f'{"".join(body)}</svg>\n'
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _git_revision(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run official LongBench/RULER subset evidence.")
    parser.add_argument("--output", default="site/official_longbench_ruler_results.json")
    parser.add_argument("--chart-dir", default="site/charts")
    parser.add_argument("--run-dir", default="/Users/ghostmesh/benchmark-runs/catalyst-kv-cache-official")
    parser.add_argument("--longbench-repo", default=str(DEFAULT_LONGBENCH_REPO))
    parser.add_argument("--ruler-repo", default=str(DEFAULT_RULER_REPO))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--execution-path", choices=("cloudflare_workers_ai", "catalyst_rain_worker"), default=DEFAULT_EXECUTION_PATH)
    parser.add_argument("--rain-worker-url", default=None)
    parser.add_argument("--rain-worker-api-key", default=None)
    parser.add_argument("--rain-context-transport", choices=("plain_prompt", "packed_tokens"), default="plain_prompt")
    parser.add_argument("--longbench-limit", type=int, default=DEFAULT_LONGBENCH_LIMIT)
    parser.add_argument("--max-input-tokens", type=int, default=DEFAULT_MAX_INPUT_TOKENS)
    parser.add_argument("--ruler-samples", type=int, default=2)
    parser.add_argument("--ruler-prompt-mode", choices=("direct", "answer_only"), default=DEFAULT_RULER_PROMPT_MODE)
    parser.add_argument("--ruler-tasks", default=",".join(DEFAULT_RULER_TASKS))
    parser.add_argument("--ruler-lengths", default=",".join(str(length) for length in DEFAULT_RULER_LENGTHS))
    parser.add_argument("--no-network", action="store_true")
    args = parser.parse_args()
    payload = run_official_longbench_ruler(
        output=args.output,
        chart_dir=args.chart_dir,
        run_dir=args.run_dir,
        longbench_repo=args.longbench_repo,
        ruler_repo=args.ruler_repo,
        model=args.model,
        longbench_limit=args.longbench_limit,
        max_input_tokens=args.max_input_tokens,
        ruler_samples=args.ruler_samples,
        ruler_prompt_mode=args.ruler_prompt_mode,
        execution_path=args.execution_path,
        rain_worker_url=args.rain_worker_url,
        rain_worker_api_key=args.rain_worker_api_key,
        rain_context_transport=args.rain_context_transport,
        ruler_tasks=tuple(item.strip() for item in args.ruler_tasks.split(",") if item.strip()),
        ruler_lengths=tuple(int(item.strip()) for item in args.ruler_lengths.split(",") if item.strip()),
        network=not args.no_network,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
