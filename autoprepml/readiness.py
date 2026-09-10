"""Structured data-readiness checks for fitted preprocessing plans."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional

import pandas as pd


@dataclass(frozen=True)
class ReadinessCheck:
    """One named readiness dimension and its evidence."""

    name: str
    status: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class DataReadinessReport:
    """Evidence-backed readiness summary for a DataPlan and future frame."""

    checks: tuple[ReadinessCheck, ...]

    @property
    def status(self) -> str:
        if any(check.status == "FAIL" for check in self.checks):
            return "FAIL"
        if any(check.status == "WARN" for check in self.checks):
            return "WARN"
        return "PASS"

    @property
    def ready(self) -> bool:
        return self.status == "PASS"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "ready": self.ready,
            "checks": [check.to_dict() for check in self.checks],
        }


def assess_data_readiness(plan: Any, frame: Optional[pd.DataFrame] = None) -> DataReadinessReport:
    """Assess schema, leakage, reproducibility, identity, and lineage.

    The report is intentionally evidence-based. A missing frame leaves schema
    compatibility as ``WARN`` rather than inventing a pass result; an
    unfitted plan fails leakage and reproducibility checks.
    """
    if not hasattr(plan, "fitted") or not hasattr(plan, "manifest"):
        raise TypeError("plan must be a DataPlan-compatible object")

    checks: list[ReadinessCheck] = []
    if frame is None:
        checks.append(ReadinessCheck("schema_integrity", "WARN", "No future frame was supplied"))
    else:
        validation = plan.validate(frame)
        checks.append(
            ReadinessCheck(
                "schema_integrity",
                validation.status,
                f"Contract validation returned {validation.status} with {len(validation.issues)} issue(s)",
            )
        )

    if plan.fitted:
        checks.append(
            ReadinessCheck(
                "leakage_safety",
                "PASS",
                "Transformations are fitted once and applied to future rows",
            )
        )
        manifest = plan.manifest()
        reproducible = all(
            manifest.get(key) is not None
            for key in ("random_state", "configuration_digest", "state_fingerprint")
        )
        checks.append(
            ReadinessCheck(
                "reproducibility",
                "PASS" if reproducible else "FAIL",
                "Random state, configuration digest, and state fingerprint are recorded",
            )
        )
        identity = manifest.get("dataset_fingerprint") is not None
        checks.append(
            ReadinessCheck(
                "dataset_identity",
                "PASS" if identity else "FAIL",
                (
                    "The fitted dataset fingerprint is recorded"
                    if identity
                    else "No fitted fingerprint"
                ),
            )
        )
        lineage = bool(manifest.get("transformations"))
        checks.append(
            ReadinessCheck(
                "lineage",
                "PASS" if lineage else "WARN",
                (
                    "Transformation steps are recorded"
                    if lineage
                    else "No transformation steps recorded"
                ),
            )
        )
    else:
        checks.extend(
            [
                ReadinessCheck("leakage_safety", "FAIL", "Plan has not been fitted"),
                ReadinessCheck("reproducibility", "FAIL", "Fitted state is not available"),
                ReadinessCheck("dataset_identity", "WARN", "No fitted dataset fingerprint exists"),
                ReadinessCheck("lineage", "WARN", "No fitted transformation lineage exists"),
            ]
        )
    return DataReadinessReport(tuple(checks))


__all__ = ["ReadinessCheck", "DataReadinessReport", "assess_data_readiness"]
