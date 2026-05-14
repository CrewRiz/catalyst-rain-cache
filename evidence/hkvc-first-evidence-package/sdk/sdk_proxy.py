from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable


class CatalystSDKFeatureUnavailable(RuntimeError):
    """Raised when the installed catalyst-brain SDK lacks a private feature."""


SDK_MODULE_CANDIDATES = (
    "catalyst_brain.hkvc",
    "catalyst_brain.evidence",
    "catalyst_brain",
)


def load_sdk_feature(name: str, *, modules: Iterable[str] = SDK_MODULE_CANDIDATES) -> Any:
    errors: list[str] = []
    for module_name in modules:
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{module_name}: {exc}")
            continue
        if hasattr(module, name):
            return getattr(module, name)
    searched = ", ".join(modules)
    details = "; ".join(errors) if errors else "no candidate exported the feature"
    raise CatalystSDKFeatureUnavailable(
        f"{name} is provided by the catalyst-brain SDK, not this public adapter. "
        f"Install a catalyst-brain build that exports it. Searched: {searched}. {details}"
    )


def call_sdk_feature(name: str, *args: Any, **kwargs: Any) -> Any:
    feature = load_sdk_feature(name)
    return feature(*args, **kwargs)
