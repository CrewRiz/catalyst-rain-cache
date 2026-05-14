from __future__ import annotations

import hashlib
import json
import math
import numbers
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

import catalyst_hdc as hdc
from catalyst_brain.rain import RainPayload, to_header

from catalyst_kv_cache.license import assert_research_use


class CatalystKVError(RuntimeError):
    """Base error for Catalyst KV-cache adapter failures."""


@dataclass(frozen=True)
class CatalystKVConfig:
    """Configuration for the Catalyst KV-cache adapter.

    mode:
        "passthrough" returns original key/value states from update(...), which
        preserves model behavior while recording Catalyst state. "refs" returns
        compact cache references for stacks that opt into Catalyst-managed
        materialization.
    """

    dim: int = 4096
    mode: str = "passthrough"
    purpose: str = "research"
    retain_last_passthrough: bool = False
    max_preview_chars: int = 80

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.mode not in {"passthrough", "refs"}:
            raise ValueError("mode must be 'passthrough' or 'refs'")
        assert_research_use(self.purpose)


@dataclass(frozen=True)
class CacheRecord:
    layer_idx: int
    position: int
    token_count: int
    key_fingerprint: str
    value_fingerprint: str
    key_bytes_estimate: int
    value_bytes_estimate: int
    kv_ref: str
    created_at: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class CatalystKVCache:
    """Drop-in research adapter for fixed-state Catalyst KV-cache experiments."""

    def __init__(self, config: CatalystKVConfig | None = None) -> None:
        self.config = config or CatalystKVConfig()
        self._kv = hdc.PyHKVC(self.config.dim)
        self._layers: dict[int, list[CacheRecord]] = {}
        self._world_vector: list[float] | None = None
        self._position_counter = 0
        self._last_passthrough: dict[int, tuple[Any, Any]] = {}

    @classmethod
    def from_transformers(
        cls,
        *,
        dim: int = 4096,
        mode: str = "passthrough",
        purpose: str = "research",
    ) -> "CatalystKVCache":
        """Create a Hugging Face style cache adapter."""
        return cls(CatalystKVConfig(dim=dim, mode=mode, purpose=purpose))

    def update(
        self,
        key_states: Any,
        value_states: Any,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[Any, Any] | dict[str, Any]:
        """Hugging Face style cache update hook.

        In passthrough mode, this returns `(key_states, value_states)` so model
        behavior stays unchanged. In refs mode, it returns a compact record
        dictionary with `kv_ref` and fingerprints.
        """
        kwargs = cache_kwargs or {}
        record = self.append(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            position=_coerce_int(kwargs.get("position"), self.get_seq_length(layer_idx)),
            token_count=_coerce_int(kwargs.get("token_count"), _infer_token_count(key_states)),
            metadata={k: v for k, v in kwargs.items() if k not in {"position", "token_count"}},
        )
        if self.config.retain_last_passthrough:
            self._last_passthrough[layer_idx] = (key_states, value_states)
        if self.config.mode == "refs":
            return record.to_dict()
        return key_states, value_states

    def append(
        self,
        *,
        layer_idx: int,
        key_states: Any,
        value_states: Any,
        position: int | None = None,
        token_count: int = 1,
        metadata: dict[str, Any] | None = None,
    ) -> CacheRecord:
        if layer_idx < 0:
            raise ValueError("layer_idx must be non-negative")
        layer_records = self._layers.setdefault(layer_idx, [])
        position = len(layer_records) if position is None else position
        key_fp, key_hv, key_bytes = _fingerprint_hv(key_states, self.config.dim)
        value_fp, value_hv, value_bytes = _fingerprint_hv(value_states, self.config.dim)
        record_fp = _digest_text(f"{layer_idx}:{position}:{key_fp}:{value_fp}")
        kv_ref = f"catalyst-kv://layer/{layer_idx}/position/{position}/{record_fp}"
        record = CacheRecord(
            layer_idx=layer_idx,
            position=position,
            token_count=max(1, token_count),
            key_fingerprint=key_fp,
            value_fingerprint=value_fp,
            key_bytes_estimate=key_bytes,
            value_bytes_estimate=value_bytes,
            kv_ref=kv_ref,
            created_at=time.time(),
            metadata=dict(metadata or {}),
        )
        payload = json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
        store_position = self._next_position()
        self._kv.store(kv_ref, payload, store_position)
        self._kv.store(f"layer:{layer_idx}:key:{key_fp}", kv_ref, self._next_position())
        self._kv.store(f"layer:{layer_idx}:value:{value_fp}", kv_ref, self._next_position())
        layer_records.append(record)
        self._absorb(key_hv)
        self._absorb(value_hv)
        return record

    def lookup(self, *, layer_idx: int, query: Any) -> dict[str, Any] | None:
        """Return a compact cache record for an exact Catalyst fingerprint hit."""
        query_fp, _, _ = _fingerprint_hv(query, self.config.dim)
        ref, confidence = self._kv.query(f"layer:{layer_idx}:key:{query_fp}")
        if not ref or confidence <= 0.0:
            ref, confidence = self._kv.query(f"layer:{layer_idx}:value:{query_fp}")
        if not ref or confidence <= 0.0:
            return None
        return self.materialize(ref)

    def materialize(self, kv_ref: str) -> dict[str, Any] | None:
        """Fetch a compact cache record by reference.

        This returns metadata/fingerprints, not original tensors. Production
        materialization strategies are commercial integration work.
        """
        encoded, confidence = self._kv.query(kv_ref)
        if not encoded or confidence <= 0.0:
            return None
        return json.loads(encoded)

    def get_seq_length(self, layer_idx: int | None = 0) -> int:
        if layer_idx is None:
            return max((sum(record.token_count for record in records) for records in self._layers.values()), default=0)
        return sum(record.token_count for record in self._layers.get(layer_idx, []))

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        del new_seq_length
        return self.get_seq_length(layer_idx)

    def get_max_length(self) -> None:
        return None

    def reorder_cache(self, beam_idx: Any) -> "CatalystKVCache":
        del beam_idx
        return self

    def reset(self) -> None:
        self._kv = hdc.PyHKVC(self.config.dim)
        self._layers.clear()
        self._world_vector = None
        self._position_counter = 0
        self._last_passthrough.clear()

    def layer_records(self, layer_idx: int) -> list[dict[str, Any]]:
        return [record.to_dict() for record in self._layers.get(layer_idx, [])]

    def to_rain_header(self, *, agent_id: str = "catalyst-kv-cache") -> str:
        vector = self._world_vector or hdc.hv_hash_string("empty-catalyst-kv-cache", self.config.dim)
        payload = RainPayload(
            agent_id=agent_id,
            dim=self.config.dim,
            world_vector=vector,
            config={
                "mode": self.config.mode,
                "layers": len(self._layers),
                "records": sum(len(records) for records in self._layers.values()),
                "purpose": self.config.purpose,
            },
        )
        return to_header(payload)

    def compression_report(self) -> dict[str, Any]:
        records = [record for layer in self._layers.values() for record in layer]
        original_bytes = sum(record.key_bytes_estimate + record.value_bytes_estimate for record in records)
        compact_records = [record.to_dict() for record in records]
        compact_bytes = len(json.dumps(compact_records, separators=(",", ":"), sort_keys=True).encode("utf-8"))
        rain_header_bytes = len(self.to_rain_header().encode("utf-8"))
        total_compact = compact_bytes + rain_header_bytes
        saved_pct = 0.0 if original_bytes <= 0 else max(0.0, 100.0 * (1.0 - total_compact / original_bytes))
        return {
            "records": len(records),
            "layers": len(self._layers),
            "mode": self.config.mode,
            "original_bytes_estimate": original_bytes,
            "compact_record_bytes": compact_bytes,
            "rain_header_bytes": rain_header_bytes,
            "total_compact_bytes": total_compact,
            "saved_pct": round(saved_pct, 4),
            "commercial_contact": "hello@strategic-innovations.ai",
        }

    def _next_position(self) -> int:
        position = self._position_counter
        self._position_counter += 1
        return position

    def _absorb(self, vector: list[float]) -> None:
        if self._world_vector is None:
            self._world_vector = vector
        else:
            self._world_vector = hdc.hdc_bundle(self._world_vector, vector)


def _coerce_int(value: Any, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _infer_token_count(value: Any) -> int:
    shape = getattr(value, "shape", None)
    if shape is not None and len(shape) >= 2:
        try:
            return max(1, int(shape[-2]))
        except (TypeError, ValueError):
            return 1
    if isinstance(value, (list, tuple)) and value:
        first = value[0]
        if isinstance(first, (list, tuple)):
            return len(value)
    return 1


def _fingerprint_hv(value: Any, dim: int) -> tuple[str, list[float], int]:
    stable = _stable_bytes(value)
    digest = hashlib.blake2b(stable, digest_size=16).hexdigest()
    return digest, hdc.hv_hash_string(digest, dim), len(stable)


def _digest_text(value: str) -> str:
    return hashlib.blake2b(value.encode("utf-8"), digest_size=8).hexdigest()


def _stable_bytes(value: Any) -> bytes:
    floats = list(_iter_floats(value, limit=8192))
    if floats:
        payload = {
            "type": type(value).__name__,
            "shape": _shape_of(value),
            "sample": [round(float(item), 8) for item in floats],
        }
        return json.dumps(payload, separators=(",", ":"), sort_keys=True).encode("utf-8")
    try:
        return json.dumps(value, separators=(",", ":"), sort_keys=True, default=str).encode("utf-8")
    except TypeError:
        return repr(value).encode("utf-8")


def _iter_floats(value: Any, *, limit: int) -> Iterable[float]:
    if limit <= 0:
        return
    if isinstance(value, numbers.Real) and math.isfinite(float(value)):
        yield float(value)
        return
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        try:
            flat = value.detach().cpu().flatten()
            for item in flat[:limit].tolist():
                if isinstance(item, numbers.Real) and math.isfinite(float(item)):
                    yield float(item)
            return
        except Exception:
            pass
    if hasattr(value, "ravel") and hasattr(value, "tolist"):
        try:
            for item in value.ravel()[:limit].tolist():
                if isinstance(item, numbers.Real) and math.isfinite(float(item)):
                    yield float(item)
            return
        except Exception:
            pass
    if isinstance(value, (list, tuple)):
        remaining = limit
        for item in value:
            if remaining <= 0:
                break
            emitted = 0
            for number in _iter_floats(item, limit=remaining):
                emitted += 1
                yield number
            remaining -= emitted


def _shape_of(value: Any) -> tuple[int, ...] | None:
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return tuple(int(item) for item in shape)
        except (TypeError, ValueError):
            return None
    if isinstance(value, (list, tuple)):
        dims: list[int] = []
        current = value
        while isinstance(current, (list, tuple)):
            dims.append(len(current))
            current = current[0] if current else []
        return tuple(dims)
    return None
