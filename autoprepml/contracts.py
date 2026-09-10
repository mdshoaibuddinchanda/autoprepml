"""Lightweight schema contracts and structured validation reports."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import pandas as pd

from .exceptions import ContractError, IntegrationError, ValidationError
from .fingerprints import schema_fingerprint


def _safe_value(value: Any) -> Any:
    """Convert values used in manifests to JSON-compatible primitives."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return str(value)


@dataclass(frozen=True)
class ColumnContract:
    """Expected properties for one input column."""

    name: str
    expected_dtype: str
    semantic_role: str = "feature"
    required: bool = True
    nullable: bool = True
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    allowed_categories: Optional[tuple[Any, ...]] = None
    unique: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""
        result = asdict(self)
        if self.allowed_categories is not None:
            result["allowed_categories"] = [_safe_value(value) for value in self.allowed_categories]
        return result


@dataclass(frozen=True)
class ValidationIssue:
    """One structured contract validation finding."""

    code: str
    severity: str
    column: Optional[str]
    message: str
    expected: Any = None
    observed: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Structured validation output with PASS, WARN, or FAIL status."""

    mode: str
    issues: tuple[ValidationIssue, ...] = field(default_factory=tuple)

    @property
    def status(self) -> str:
        """Return the highest severity present in the report."""
        if any(issue.severity == "FAIL" for issue in self.issues):
            return "FAIL"
        if any(issue.severity == "WARN" for issue in self.issues):
            return "WARN"
        return "PASS"

    @property
    def compatible(self) -> bool:
        """Whether the report contains no blocking failures."""
        return self.status != "FAIL"

    def raise_for_error(self) -> None:
        """Raise :class:`ContractError` when the report is not compatible."""
        if not self.compatible:
            details = "; ".join(issue.message for issue in self.issues if issue.severity == "FAIL")
            raise ContractError(details or "Data does not satisfy the contract")

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "compatible": self.compatible,
            "issues": [issue.to_dict() for issue in self.issues],
        }


@dataclass(frozen=True)
class DataContract:
    """A small, explicit contract for a tabular dataset.

    Contracts intentionally cover common compatibility checks rather than
    attempting to replace a full data-observability framework. Unknown feature
    categories are warnings in compatible mode and failures in strict mode.
    """

    columns: tuple[ColumnContract, ...]
    target: Optional[str] = None
    task: Optional[str] = None
    require_column_order: bool = False

    @property
    def feature_columns(self) -> tuple[str, ...]:
        return tuple(column.name for column in self.columns if column.semantic_role != "target")

    @classmethod
    def infer(
        cls,
        frame: pd.DataFrame,
        target: Optional[str] = None,
        task: Optional[str] = None,
        max_categories: int = 1000,
    ) -> "DataContract":
        """Infer a conservative contract from a training DataFrame."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if frame.columns.has_duplicates:
            raise ContractError("DataFrame columns must be unique")
        if target is not None and target not in frame.columns:
            raise ContractError(f"Target column '{target}' not found in DataFrame")
        if task not in (None, "classification", "regression"):
            raise ContractError("task must be 'classification', 'regression', or None")
        if not isinstance(max_categories, int) or isinstance(max_categories, bool) or max_categories <= 0:
            raise ValueError("max_categories must be a positive integer")

        contracts = []
        for name in frame.columns:
            series = frame[name]
            role = "target" if name == target else "feature"
            categories = None
            if role == "feature" and (
                pd.api.types.is_object_dtype(series)
                or isinstance(series.dtype, pd.CategoricalDtype)
                or pd.api.types.is_bool_dtype(series)
            ):
                unique = series.dropna().unique().tolist()
                if len(unique) <= max_categories:
                    categories = tuple(sorted(unique, key=lambda value: repr(value)))
            contracts.append(
                ColumnContract(
                    name=str(name),
                    expected_dtype=str(series.dtype),
                    semantic_role=role,
                    nullable=role != "target",
                    allowed_categories=categories,
                )
            )
        return cls(tuple(contracts), target=target, task=task)

    def to_dict(self) -> dict[str, Any]:
        return {
            "columns": [column.to_dict() for column in self.columns],
            "target": self.target,
            "task": self.task,
            "require_column_order": self.require_column_order,
            "schema_fingerprint": self.schema_fingerprint,
        }

    @property
    def schema_fingerprint(self) -> str:
        """Return a digest of contract structure without data values."""
        frame = pd.DataFrame(
            {column.name: pd.Series(dtype=column.expected_dtype) for column in self.columns}
        )
        return schema_fingerprint(frame, target=self.target)

    def _issue(
        self,
        issues: list[ValidationIssue],
        code: str,
        severity: str,
        column: Optional[str],
        message: str,
        expected: Any = None,
        observed: Any = None,
    ) -> None:
        issues.append(ValidationIssue(code, severity, column, message, expected, observed))

    def validate(
        self,
        frame: pd.DataFrame,
        mode: str = "compatible",
        include_target: bool = False,
    ) -> ValidationReport:
        """Validate a frame in ``compatible`` or ``strict`` mode."""
        if not isinstance(frame, pd.DataFrame):
            raise ValidationError("frame must be a pandas DataFrame")
        mode = mode.lower().strip() if isinstance(mode, str) else mode
        if mode not in {"compatible", "strict"}:
            raise ValidationError("mode must be 'compatible' or 'strict'")
        expected = [
            column
            for column in self.columns
            if include_target or column.semantic_role != "target"
        ]
        expected_names = [column.name for column in expected]
        issues: list[ValidationIssue] = []

        if frame.columns.has_duplicates:
            self._issue(
                issues,
                "duplicate_columns",
                "FAIL",
                None,
                "Input contains duplicate column names",
                expected="unique column names",
                observed=list(frame.columns),
            )

        missing = [name for name in expected_names if name not in frame.columns]
        for name in missing:
            contract = next(column for column in expected if column.name == name)
            severity = "FAIL" if contract.required else "WARN"
            self._issue(
                issues,
                "missing_required_column" if contract.required else "missing_optional_column",
                severity,
                name,
                f"Required column '{name}' is missing" if contract.required else f"Optional column '{name}' is missing",
                expected=contract.expected_dtype,
            )

        extras = [name for name in frame.columns if name not in expected_names]
        for name in extras:
            severity = "FAIL" if mode == "strict" else "WARN"
            self._issue(
                issues,
                "unexpected_column",
                severity,
                str(name),
                f"Unexpected column '{name}' is present",
                expected=expected_names,
                observed=str(name),
            )

        present_names = [name for name in expected_names if name in frame.columns]
        actual_order = [name for name in frame.columns if name in expected_names]
        if present_names != actual_order and self.require_column_order:
            severity = "FAIL" if mode == "strict" else "WARN"
            self._issue(
                issues,
                "column_order_mismatch",
                severity,
                None,
                "Input feature columns are not in the contract order",
                expected=present_names,
                observed=actual_order,
            )

        for contract in expected:
            if contract.name not in frame.columns:
                continue
            series = frame[contract.name]
            observed_dtype = str(series.dtype)
            if observed_dtype != contract.expected_dtype:
                severity = "FAIL" if mode == "strict" else "WARN"
                self._issue(
                    issues,
                    "dtype_mismatch",
                    severity,
                    contract.name,
                    f"Column '{contract.name}' has dtype {observed_dtype}; expected {contract.expected_dtype}",
                    expected=contract.expected_dtype,
                    observed=observed_dtype,
                )
            null_count = int(series.isna().sum())
            if null_count and not contract.nullable:
                self._issue(
                    issues,
                    "null_not_allowed",
                    "FAIL",
                    contract.name,
                    f"Column '{contract.name}' contains {null_count} null values",
                    expected="no null values",
                    observed=null_count,
                )
            if contract.minimum is not None or contract.maximum is not None:
                numeric = pd.to_numeric(series, errors="coerce")
                if contract.minimum is not None and (numeric < contract.minimum).any():
                    self._issue(
                        issues,
                        "below_minimum",
                        "FAIL",
                        contract.name,
                        f"Column '{contract.name}' contains values below the minimum",
                        expected=contract.minimum,
                        observed=float(numeric.min()),
                    )
                if contract.maximum is not None and (numeric > contract.maximum).any():
                    self._issue(
                        issues,
                        "above_maximum",
                        "FAIL",
                        contract.name,
                        f"Column '{contract.name}' contains values above the maximum",
                        expected=contract.maximum,
                        observed=float(numeric.max()),
                    )
            if contract.allowed_categories is not None:
                observed = set(series.dropna().tolist())
                allowed = set(contract.allowed_categories)
                unknown = sorted(observed - allowed, key=repr)
                if unknown:
                    severity = "FAIL" if mode == "strict" else "WARN"
                    self._issue(
                        issues,
                        "unknown_category",
                        severity,
                        contract.name,
                        f"Column '{contract.name}' contains unseen categories",
                        expected=list(contract.allowed_categories),
                        observed=[_safe_value(value) for value in unknown],
                    )
            if contract.unique and series.duplicated().any():
                self._issue(
                    issues,
                    "unique_constraint",
                    "FAIL",
                    contract.name,
                    f"Column '{contract.name}' contains duplicate values",
                    expected="unique values",
                    observed=int(series.duplicated().sum()),
                )
        return ValidationReport(mode=mode, issues=tuple(issues))

    def to_pandera(self):
        """Export common checks to Pandera when the optional dependency exists."""
        try:
            import pandera as pa
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise IntegrationError("Pandera is required for DataContract.to_pandera()") from exc

        columns = {}
        for contract in self.columns:
            checks = []
            if contract.minimum is not None:
                checks.append(pa.Check.ge(contract.minimum))
            if contract.maximum is not None:
                checks.append(pa.Check.le(contract.maximum))
            if contract.allowed_categories is not None:
                checks.append(pa.Check.isin(list(contract.allowed_categories)))
            columns[contract.name] = pa.Column(
                contract.expected_dtype,
                checks=checks,
                nullable=contract.nullable,
                required=contract.required,
                unique=contract.unique,
            )
        return pa.DataFrameSchema(columns, strict=self.require_column_order)


__all__ = ["ColumnContract", "ValidationIssue", "ValidationReport", "DataContract"]
