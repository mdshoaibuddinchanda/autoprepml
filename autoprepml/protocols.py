"""Typing protocols for compatible preprocessing components."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class PrepProtocol(Protocol):
    """Minimal lifecycle shared by fitted preprocessing components."""

    def inspect(self) -> dict[str, Any]:
        """Return schema and configuration metadata without raw records."""

    def fit(self, data: Any) -> "PrepProtocol":
        """Acquire or learn state from training data."""

    def transform(self, data: Any) -> Any:
        """Apply acquired state without learning new statistics."""

    def validate(self, data: Any, mode: str = "compatible") -> Any:
        """Return structured compatibility findings."""

    def report(self) -> Any:
        """Return a modality-specific report."""


__all__ = ["PrepProtocol"]
