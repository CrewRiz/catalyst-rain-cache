from __future__ import annotations

COMMERCIAL_CONTACT = "hello@strategic-innovations.ai"

COMMERCIAL_TERMS = {
    "commercial",
    "enterprise",
    "hosted",
    "pilot",
    "production",
    "revenue",
    "saas",
}


class CatalystKVLicenseError(RuntimeError):
    """Raised when code explicitly declares a commercial use context."""


def assert_research_use(purpose: str = "research") -> None:
    """Reject explicitly commercial use declarations.

    This helper is not a security boundary. It exists so examples and
    integrations can make the license boundary visible in code.
    """
    lowered = purpose.lower()
    if any(term in lowered for term in COMMERCIAL_TERMS):
        raise CatalystKVLicenseError(
            "Catalyst KV Cache is research/evaluation source-available. "
            "Production, enterprise, hosted, commercial, or pilot use requires "
            f"a written license. Contact {COMMERCIAL_CONTACT}."
        )
