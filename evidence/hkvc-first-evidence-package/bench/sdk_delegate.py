from __future__ import annotations

from importlib import import_module
from typing import Any, Iterable, Sequence


class CatalystSDKFeatureUnavailable(RuntimeError):
    """Raised when an evidence generator is missing from catalyst-brain."""


SDK_MODULE_CANDIDATES = (
    "catalyst_brain.hkvc_evidence",
    "catalyst_brain.evidence",
    "catalyst_brain",
)


def load_catalyst_brain_callable(
    names: str | Sequence[str],
    *,
    modules: Iterable[str] = SDK_MODULE_CANDIDATES,
) -> Any:
    requested = (names,) if isinstance(names, str) else tuple(names)
    errors: list[str] = []
    for module_name in modules:
        try:
            module = import_module(module_name)
        except Exception as exc:  # pragma: no cover - environment dependent
            errors.append(f"{module_name}: {exc}")
            continue
        for name in requested:
            if hasattr(module, name):
                return getattr(module, name)
    searched_names = ", ".join(requested)
    searched_modules = ", ".join(modules)
    details = "; ".join(errors) if errors else "no candidate exported a requested callable"
    raise CatalystSDKFeatureUnavailable(
        "Private HKVC evidence generation lives in catalyst-brain. "
        f"Missing callable(s): {searched_names}. Searched modules: {searched_modules}. {details}"
    )


def call_catalyst_brain(names: str | Sequence[str], *args: Any, **kwargs: Any) -> Any:
    return load_catalyst_brain_callable(names)(*args, **kwargs)
