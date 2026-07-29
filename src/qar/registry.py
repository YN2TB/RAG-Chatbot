"""Name -> factory lookup, so ablations can swap components from a config file.

Every trainable task registers itself here; `configs/*.yaml` then names one via
`task: <name>` and the harness never imports it directly.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

_REGISTRIES: dict[str, dict[str, Callable[..., Any]]] = {}


def register(kind: str, name: str) -> Callable[[Callable], Callable]:
    """Decorator: `@register("task", "biencoder")` above a class or factory."""

    def wrap(fn: Callable) -> Callable:
        bucket = _REGISTRIES.setdefault(kind, {})
        if name in bucket:
            raise KeyError(f"{kind} '{name}' is already registered by {bucket[name]!r}")
        bucket[name] = fn
        return fn

    return wrap


def build(kind: str, name: str, *args: Any, **kwargs: Any) -> Any:
    bucket = _REGISTRIES.get(kind, {})
    if name not in bucket:
        known = ", ".join(sorted(bucket)) or "<none registered>"
        raise KeyError(f"unknown {kind} '{name}'. Registered: {known}")
    return bucket[name](*args, **kwargs)


def available(kind: str) -> list[str]:
    return sorted(_REGISTRIES.get(kind, {}))
