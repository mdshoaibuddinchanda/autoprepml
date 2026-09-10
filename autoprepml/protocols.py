"""Typing protocols for compatible preprocessing components."""

from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable


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


@runtime_checkable
class ExperimentRunProtocol(Protocol):
    """Common logging surface shared by local and optional trackers."""

    def log_params(self, params: dict[str, Any], **kwargs: Any) -> None:
        """Record run parameters."""

    def log_metrics(self, metrics: dict[str, float], **kwargs: Any) -> None:
        """Record numeric run metrics."""

    def log_artifact(self, path: Any, artifact_name: Optional[str] = None) -> Any:
        """Record a file artifact."""

    def log_plan(self, plan: Any, artifact_name: str = "data-plan.json") -> Any:
        """Record a plan manifest without raw data rows."""


@runtime_checkable
class ExperimentTrackerProtocol(Protocol):
    """Common tracker factory surface."""

    def start_run(self, name: str = "autoprepml-run", tags: Optional[dict[str, str]] = None) -> Any:
        """Start a context-managed experiment run."""


__all__ = ["PrepProtocol", "ExperimentRunProtocol", "ExperimentTrackerProtocol"]
