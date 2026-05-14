from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE_ROOT = REPO_ROOT / "evidence" / "hkvc-first-evidence-package"


def test_evidence_package_does_not_ship_private_algorithm_modules():
    leaked_modules = [
        "sdk/cache.py",
        "sdk/bridge.py",
        "sdk/edge_cache.py",
        "sdk/hf_cache.py",
        "sdk/rain.py",
        "sdk/manifold_attention.py",
    ]

    leaked = [module for module in leaked_modules if (EVIDENCE_ROOT / module).exists()]

    assert leaked == []


def test_benchmark_entrypoints_delegate_private_algorithms_to_catalyst_brain():
    entrypoints = [
        "bench/run_benchmarks.py",
        "bench/manifold_attention_benchmark.py",
        "bench/lossless_equivalence.py",
        "bench/lossless_scale.py",
        "bench/tier2_evidence.py",
    ]
    forbidden_snippets = [
        "sdk.cache",
        "sdk.manifold_attention",
        "sum(phi",
        "outer v",
        "softmax-moment",
        "signature_boost",
        "fixed_softmax_moment_sketch",
        "phi(q)^T",
        "promote_to_compact",
        "QuantumAttentionHead",
        "CatalystManifoldAttention",
    ]

    for entrypoint in entrypoints:
        text = (EVIDENCE_ROOT / entrypoint).read_text()
        assert "catalyst_brain" in text
        for snippet in forbidden_snippets:
            assert snippet not in text


def test_public_next_evidence_runner_does_not_disclose_private_algorithms():
    texts = [
        (EVIDENCE_ROOT / "bench/next_evidence.py").read_text(),
        (EVIDENCE_ROOT / "bench/official_longbench_ruler.py").read_text(),
        (EVIDENCE_ROOT / "bench/rain_transport_probe.py").read_text(),
    ]
    forbidden_snippets = [
        "sdk.cache",
        "sdk.manifold_attention",
        "sum(phi",
        "outer v",
        "softmax-moment",
        "signature_boost",
        "fixed_softmax_moment_sketch",
        "phi(q)^T",
        "promote_to_compact",
        "QuantumAttentionHead",
        "CatalystManifoldAttention",
    ]

    assert "CLOUDFLARE_API_TOKEN" in texts[0]
    assert "CLOUDFLARE_API_TOKEN" in texts[1]
    for text in texts:
        for snippet in forbidden_snippets:
            assert snippet not in text


def test_public_docs_and_artifacts_do_not_disclose_algorithm_internals():
    public_files = [
        "README.md",
        "site/index.html",
        "site/manifold_attention_results.json",
        "site/lossless_equivalence_results.json",
        "site/lossless_scale_results.json",
        "site/next_evidence_results.json",
        "site/official_longbench_ruler_results.json",
        "site/rain_transport_probe_results.json",
        "site/tier2_evidence_results.json",
        "site/charts/manifold_architecture.svg",
        "site/charts/rain_worker_transport.svg",
    ]
    forbidden_snippets = [
        "sum(phi",
        "outer v",
        "softmax-moment",
        "signed collision",
        "signature_boost",
        "fixed_softmax_moment_sketch",
        "phi(q)^T",
        "promote_to_compact",
        "QuantumAttentionHead",
        "CatalystManifoldAttention",
    ]

    for public_file in public_files:
        text = (EVIDENCE_ROOT / public_file).read_text()
        for snippet in forbidden_snippets:
            assert snippet not in text


def test_public_repo_does_not_expose_legacy_private_transport_surface():
    legacy_abbrev = "H" + "M" + "K"
    legacy_name = "Hypervector " + "Memory " + "Key"
    forbidden_terms = [
        legacy_abbrev,
        legacy_abbrev.lower(),
        legacy_name,
        legacy_name.lower(),
    ]
    excluded_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "__pycache__",
        "node_modules",
    }
    public_files = [
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file()
        and excluded_parts.isdisjoint(path.parts)
    ]

    leaks = []
    for public_file in public_files:
        text = public_file.read_text(encoding="utf-8", errors="ignore")
        for term in forbidden_terms:
            if term in text:
                leaks.append(str(public_file.relative_to(REPO_ROOT)))

    assert sorted(set(leaks)) == []
