"""Canonical fitted preprocessing plans for leakage-safe ML workflows."""

from __future__ import annotations

import hashlib
import json
import os

# Serialized state is integrity-checked and trusted-source-only.
import pickle  # nosec B403
import platform
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd
import joblib
from scipy import sparse

from .config import validate_config
from .contracts import DataContract, ValidationReport
from .exceptions import ArtifactError, ConfigurationError, ContractError, NotFittedError
from .fingerprints import DatasetFingerprint, fingerprint_dataframe
from .pipeline import make_preprocessing_pipeline
from .readiness import DataReadinessReport, assess_data_readiness


_ARTIFACT_FORMAT_VERSION = "1.0"
_DEFAULT_FINGERPRINT_SAMPLE_ROWS = 1_000
_OUTPUT_FORMATS = {"pandas", "sparse", "numpy"}
_DEFAULT_MAX_DENSE_ELEMENTS = 10_000_000


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_digest(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return _sha256(encoded.encode("utf-8"))


class DataPlan:
    """Infer, fit, validate, and serialize a reusable preprocessing plan.

    A plan has two states. ``infer`` creates an ``UNFITTED`` plan containing
    only schema and configuration metadata. ``fit`` learns imputation,
    category, and scaling state from training rows. ``transform`` never learns
    new statistics and returns a feature-only DataFrame in fitted column order.
    """

    def __init__(
        self,
        contract: DataContract,
        config: dict[str, Any],
        random_state: int = 42,
        fit_fingerprint: Optional[DatasetFingerprint] = None,
        fingerprint_mode: str = "full",
        fingerprint_sample_rows: int = _DEFAULT_FINGERPRINT_SAMPLE_ROWS,
        output_format: str = "pandas",
        max_dense_elements: int = _DEFAULT_MAX_DENSE_ELEMENTS,
    ) -> None:
        fingerprint_mode = (
            fingerprint_mode.lower().strip()
            if isinstance(fingerprint_mode, str)
            else fingerprint_mode
        )
        if fingerprint_mode not in {"schema", "sampled", "full"}:
            raise ValueError("fingerprint_mode must be schema, sampled, or full")
        if (
            not isinstance(fingerprint_sample_rows, int)
            or isinstance(fingerprint_sample_rows, bool)
            or fingerprint_sample_rows < 1
        ):
            raise ValueError("fingerprint_sample_rows must be a positive integer")
        output_format = (
            output_format.lower().strip() if isinstance(output_format, str) else output_format
        )
        if output_format not in _OUTPUT_FORMATS:
            raise ValueError("output_format must be pandas, sparse, or numpy")
        if (
            not isinstance(max_dense_elements, int)
            or isinstance(max_dense_elements, bool)
            or max_dense_elements < 1
        ):
            raise ValueError("max_dense_elements must be a positive integer")
        self.contract = contract
        self.config = config
        self.random_state = random_state
        self._fit_fingerprint = fit_fingerprint
        self.fingerprint_mode = fingerprint_mode
        self.fingerprint_sample_rows = fingerprint_sample_rows
        self.output_format = output_format
        self.max_dense_elements = max_dense_elements
        self._pipeline: Any = None
        self._output_columns: Optional[tuple[str, ...]] = None
        self._feature_columns = contract.feature_columns
        self._lineage: list[dict[str, Any]] = []
        self._fit_timestamp_utc: Optional[str] = None

    @classmethod
    def infer(
        cls,
        frame: pd.DataFrame,
        target: Optional[str] = None,
        task: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
        random_state: int = 42,
        fingerprint_mode: str = "full",
        fingerprint_sample_rows: int = _DEFAULT_FINGERPRINT_SAMPLE_ROWS,
        output_format: str = "pandas",
        max_dense_elements: int = _DEFAULT_MAX_DENSE_ELEMENTS,
    ) -> "DataPlan":
        """Infer an unfitted plan from training schema and configuration."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if not isinstance(random_state, int) or isinstance(random_state, bool):
            raise TypeError("random_state must be an integer")
        if any(not isinstance(column, str) for column in frame.columns):
            raise ContractError("DataPlan requires string column names")
        contract = DataContract.infer(frame, target=target, task=task)
        merged_config = validate_config(config or {})
        fingerprint_mode = (
            fingerprint_mode.lower().strip()
            if isinstance(fingerprint_mode, str)
            else fingerprint_mode
        )
        fingerprint = fingerprint_dataframe(
            frame,
            mode=fingerprint_mode,
            sample_rows=fingerprint_sample_rows,
            target=target,
        )
        return cls(
            contract,
            merged_config,
            random_state,
            fit_fingerprint=fingerprint,
            fingerprint_mode=fingerprint_mode,
            fingerprint_sample_rows=fingerprint_sample_rows,
            output_format=output_format,
            max_dense_elements=max_dense_elements,
        )

    @property
    def fitted(self) -> bool:
        """Whether learned preprocessing state is available."""
        return self._pipeline is not None

    @property
    def state(self) -> str:
        """Return ``UNFITTED`` or ``FITTED``."""
        return "FITTED" if self.fitted else "UNFITTED"

    def _require_fitted(self) -> None:
        if not self.fitted:
            raise NotFittedError("DataPlan is not fitted; call plan.fit(training_data) first")

    def inspect(self) -> dict[str, Any]:
        """Return schema and plan metadata without raw data values."""
        return {
            "state": self.state,
            "target": self.contract.target,
            "task": self.contract.task,
            "feature_columns": list(self._feature_columns),
            "output_columns": list(self._output_columns or ()),
            "contract": self.contract.to_dict(),
            "dataset_fingerprint": (
                self._fit_fingerprint.to_dict() if self._fit_fingerprint is not None else None
            ),
            "random_state": self.random_state,
            "fingerprint_mode": self.fingerprint_mode,
            "fingerprint_sample_rows": self.fingerprint_sample_rows,
            "output_format": self.output_format,
            "max_dense_elements": self.max_dense_elements,
        }

    def report(self) -> dict[str, Any]:
        """Return the current plan report for the common preparation protocol.

        Before fitting, this is an inspection report.  After fitting, it is
        the complete reproducibility manifest, including learned output
        columns and transformation lineage.
        """
        return self.manifest() if self.fitted else self.inspect()

    def readiness_report(self, frame: Optional[pd.DataFrame] = None) -> DataReadinessReport:
        """Return evidence-backed data-readiness checks for this plan."""
        return assess_data_readiness(self, frame)

    def _pipeline_kwargs(self) -> dict[str, Any]:
        cleaning = self.config.get("cleaning", {})
        categorical_strategy = cleaning.get("categorical_strategy", "mode")
        if categorical_strategy == "mode":
            categorical_strategy = "most_frequent"
        return {
            "target_col": self.contract.target,
            "scale_numeric": cleaning.get("scale_method") is not None,
            "scale_method": cleaning.get("scale_method", "standard"),
            "numeric_strategy": cleaning.get("numeric_strategy", "median"),
            "categorical_strategy": categorical_strategy,
            "encode_method": cleaning.get("encode_method", "onehot"),
        }

    def _build_lineage(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        cleaning = self.config.get("cleaning", {})
        numeric = [
            column
            for column in self._feature_columns
            if pd.api.types.is_numeric_dtype(frame[column])
            and not pd.api.types.is_bool_dtype(frame[column])
        ]
        categorical = [column for column in self._feature_columns if column not in numeric]
        lineage: list[dict[str, Any]] = []
        if numeric:
            lineage.append(
                {
                    "name": f"{cleaning.get('numeric_strategy', 'median')}_imputation",
                    "columns": numeric,
                }
            )
            if cleaning.get("scale_method") is not None:
                lineage.append(
                    {
                        "name": f"{cleaning.get('scale_method', 'standard')}_scaling",
                        "columns": numeric,
                    }
                )
        if categorical:
            lineage.append(
                {
                    "name": f"{cleaning.get('encode_method', 'onehot')}_encoding",
                    "columns": categorical,
                }
            )
        return lineage

    def fit(self, frame: pd.DataFrame) -> "DataPlan":
        """Fit all learned transformations on the supplied training frame."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        if any(not isinstance(column, str) for column in frame.columns):
            raise ContractError("DataPlan requires string column names")
        report = self.contract.validate(frame, mode="compatible", include_target=True)
        report.raise_for_error()
        if self.contract.target and frame[self.contract.target].isna().any():
            raise ContractError("Training target contains missing values")
        features = frame.loc[:, self._feature_columns]
        if features.shape[1] == 0:
            raise ContractError("DataPlan requires at least one feature column")
        try:
            pipeline = make_preprocessing_pipeline(frame, **self._pipeline_kwargs())
            pipeline.fit(frame)
            output_columns = tuple(str(value) for value in pipeline.get_feature_names_out())
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"Could not fit DataPlan preprocessing: {exc}") from exc
        self._pipeline = pipeline
        self._output_columns = output_columns
        self._lineage = self._build_lineage(frame)
        self._fit_fingerprint = fingerprint_dataframe(
            frame,
            mode=self.fingerprint_mode,
            sample_rows=self.fingerprint_sample_rows,
            target=self.contract.target,
        )
        self._fit_timestamp_utc = datetime.now(timezone.utc).isoformat()
        return self

    def transform(self, frame: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray[Any, Any]]:
        """Transform compatible data using fitted state only.

        The default ``pandas`` output preserves the historical DataFrame API.
        Use ``sparse`` to receive a pandas sparse DataFrame without a dense
        allocation, or ``numpy`` for a dense array. Dense conversion is
        refused when it exceeds ``max_dense_elements``.
        """
        self._require_fitted()
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        report = self.contract.validate(frame, mode="compatible", include_target=False)
        report.raise_for_error()
        features = frame.loc[:, self._feature_columns]
        transformed = self._pipeline.transform(features)
        output_columns = list(self._output_columns or ())
        if sparse.issparse(transformed):
            matrix = transformed.tocsr()
            if self.output_format == "sparse":
                return pd.DataFrame.sparse.from_spmatrix(
                    matrix, index=frame.index, columns=output_columns
                )
            if matrix.shape[0] * matrix.shape[1] > self.max_dense_elements:
                raise MemoryError(
                    "Dense DataPlan output exceeds max_dense_elements; "
                    "use output_format='sparse' or increase the explicit limit"
                )
            transformed = matrix.toarray()
        if self.output_format == "numpy":
            return np.asarray(transformed)
        if self.output_format == "sparse":
            return pd.DataFrame.sparse.from_spmatrix(
                sparse.csr_matrix(np.asarray(transformed)),
                index=frame.index,
                columns=output_columns,
            )
        return pd.DataFrame(np.asarray(transformed), index=frame.index, columns=output_columns)

    def fit_transform(self, frame: pd.DataFrame) -> Union[pd.DataFrame, np.ndarray[Any, Any]]:
        """Fit on training data and immediately transform it."""
        return self.fit(frame).transform(frame)

    def validate(self, frame: pd.DataFrame, mode: str = "compatible") -> ValidationReport:
        """Return a structured compatibility report for future data."""
        return self.contract.validate(frame, mode=mode, include_target=False)

    def fit_resample(
        self,
        features: pd.DataFrame,
        target: Sequence[Any],
        method: Optional[str] = None,
    ) -> tuple[pd.DataFrame, pd.Series]:
        """Resample training rows after transformation; never used by ``transform``."""
        self._require_fitted()
        if not isinstance(features, pd.DataFrame):
            raise TypeError("features must be a pandas DataFrame")
        target_series = pd.Series(target).reset_index(drop=True)
        if len(features) != len(target_series):
            raise ValueError("features and target must contain the same number of rows")
        selected = (method or self.config.get("cleaning", {}).get("balance_method", "none")).lower()
        if selected in {"none", "disabled"}:
            return self.transform(features), target_series.copy()
        if selected not in {"oversample", "undersample", "smote"}:
            raise ValueError("method must be none, oversample, undersample, or smote")
        try:
            from imblearn.over_sampling import RandomOverSampler, SMOTE
            from imblearn.under_sampling import RandomUnderSampler
        except ImportError as exc:  # pragma: no cover - required runtime dependency
            raise ConfigurationError("imbalanced-learn is required for fit_resample") from exc
        transformed = self.transform(features)
        samplers = {
            "oversample": RandomOverSampler(random_state=self.random_state),
            "undersample": RandomUnderSampler(random_state=self.random_state),
            "smote": SMOTE(random_state=self.random_state),
        }
        try:
            values, sampled_target = samplers[selected].fit_resample(transformed, target_series)
        except (TypeError, ValueError) as exc:
            raise ContractError(f"Could not resample training data with {selected}: {exc}") from exc
        if sparse.issparse(values):
            values = values.toarray()
        return (
            pd.DataFrame(values, columns=list(self._output_columns or ())),
            pd.Series(sampled_target, name=target_series.name),
        )

    def _state_payload(self) -> bytes:
        self._require_fitted()
        return pickle.dumps(
            {
                "pipeline": self._pipeline,
                "contract": self.contract,
                "config": self.config,
                "random_state": self.random_state,
                "fit_fingerprint": self._fit_fingerprint,
                "fingerprint_mode": self.fingerprint_mode,
                "fingerprint_sample_rows": self.fingerprint_sample_rows,
                "output_format": self.output_format,
                "max_dense_elements": self.max_dense_elements,
                "output_columns": self._output_columns,
                "feature_columns": self._feature_columns,
                "lineage": self._lineage,
                "fit_timestamp_utc": self._fit_timestamp_utc,
            },
            protocol=5,
        )

    @property
    def state_fingerprint(self) -> str:
        """Digest fitted state for reproducibility and invariant tests."""
        payload = {
            "pipeline": joblib.hash(self._pipeline),
            "contract": self.contract.to_dict(),
            "config": self.config,
            "random_state": self.random_state,
            "fingerprint_mode": self.fingerprint_mode,
            "fingerprint_sample_rows": self.fingerprint_sample_rows,
            "output_format": self.output_format,
            "max_dense_elements": self.max_dense_elements,
            "output_columns": self._output_columns,
            "lineage": self._lineage,
        }
        return _json_digest(payload)

    def manifest(self) -> dict[str, Any]:
        """Return machine-readable transformation lineage and provenance."""
        self._require_fitted()
        import autoprepml
        import sklearn

        return {
            "artifact_format_version": _ARTIFACT_FORMAT_VERSION,
            "autoprepml_version": autoprepml.__version__,
            "python_version": platform.python_version(),
            "dependency_versions": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "task": self.contract.task,
            "target": self.contract.target,
            "dataset_fingerprint": (
                self._fit_fingerprint.to_dict() if self._fit_fingerprint else None
            ),
            "schema_fingerprint": self.contract.schema_fingerprint,
            "configuration_digest": _json_digest(self.config),
            "random_state": self.random_state,
            "fingerprint_mode": self.fingerprint_mode,
            "fingerprint_sample_rows": self.fingerprint_sample_rows,
            "output_format": self.output_format,
            "max_dense_elements": self.max_dense_elements,
            "fit_timestamp_utc": self._fit_timestamp_utc,
            "input_columns": list(self._feature_columns),
            "output_columns": list(self._output_columns or ()),
            "transformations": self._lineage,
            "warnings": [],
            "state_fingerprint": self.state_fingerprint,
        }

    def save(self, path: Union[str, os.PathLike[str]]) -> Path:
        """Atomically write a versioned ``.apml`` artifact."""
        self._require_fitted()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        state = self._state_payload()
        manifest = self.manifest()
        manifest["state_sha256"] = _sha256(state)
        temporary_path: Optional[Path] = None
        try:
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=f".{destination.name}-", suffix=".tmp", dir=destination.parent
            )
            os.close(descriptor)
            temporary_path = Path(temporary_name)
            with zipfile.ZipFile(temporary_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr("manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
                archive.writestr("state.pkl", state)
            os.replace(temporary_path, destination)
        except (OSError, zipfile.BadZipFile) as exc:
            raise ArtifactError(f"Could not save DataPlan artifact: {destination}") from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return destination

    @classmethod
    def load(cls, path: Union[str, os.PathLike[str]]) -> "DataPlan":
        """Load and integrity-check an artifact.

        Pickle can execute arbitrary code. Only load artifacts from trusted
        sources; the checksum detects corruption but does not make pickle safe.
        """
        source = Path(path)
        try:
            with zipfile.ZipFile(source, "r") as archive:
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                state = archive.read("state.pkl")
        except (OSError, KeyError, json.JSONDecodeError, zipfile.BadZipFile) as exc:
            raise ArtifactError(f"Invalid DataPlan artifact: {source}") from exc
        if (
            manifest.get("artifact_format_version", "").split(".")[0]
            != _ARTIFACT_FORMAT_VERSION.split(".")[0]
        ):
            raise ArtifactError(
                f"Unsupported DataPlan artifact format: {manifest.get('artifact_format_version')}"
            )
        if manifest.get("state_sha256") != _sha256(state):
            raise ArtifactError("DataPlan artifact checksum mismatch")
        try:
            # Callers must trust artifacts; checksum validation detects
            # corruption but cannot make pickle safe.
            payload = pickle.loads(state)  # nosec B301
            required = {
                "pipeline",
                "contract",
                "config",
                "random_state",
                "fit_fingerprint",
                "output_columns",
                "feature_columns",
                "lineage",
                "fit_timestamp_utc",
            }
            if not isinstance(payload, dict) or not required.issubset(payload):
                raise ValueError("missing state fields")
            plan = cls(
                payload["contract"],
                payload["config"],
                payload["random_state"],
                payload["fit_fingerprint"],
                payload.get("fingerprint_mode", "full"),
                payload.get("fingerprint_sample_rows", _DEFAULT_FINGERPRINT_SAMPLE_ROWS),
                payload.get("output_format", "pandas"),
                payload.get("max_dense_elements", _DEFAULT_MAX_DENSE_ELEMENTS),
            )
            plan._pipeline = payload["pipeline"]
            plan._output_columns = tuple(payload["output_columns"])
            plan._feature_columns = tuple(payload["feature_columns"])
            plan._lineage = list(payload["lineage"])
            plan._fit_timestamp_utc = payload["fit_timestamp_utc"]
            return plan
        except Exception as exc:
            if isinstance(exc, ArtifactError):
                raise
            raise ArtifactError(f"Could not load DataPlan state from {source}") from exc


__all__ = ["DataPlan"]
