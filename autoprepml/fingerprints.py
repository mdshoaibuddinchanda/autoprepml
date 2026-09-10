"""Deterministic fingerprints for dataset schemas and content.

Fingerprints identify data without placing raw records in reports or manifests.
The sampled mode is an efficient diagnostic digest, not a cryptographically
complete representation of the full dataset.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


def _json_default(value: Any) -> Any:
    """Convert uncommon pandas and NumPy values to stable text."""
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    return str(value)


def _canonical_dtype(dtype: Any) -> str:
    """Return a stable dtype label across pandas 2 and pandas 3.

    Pandas 3 defaults plain text columns to ``str`` while pandas 2 commonly
    reports the same columns as ``object``.  Both represent text at this
    library boundary, so fingerprints use one portable label.
    """
    label = str(dtype)
    if label in {"object", "str", "string", "string[python]", "string[pyarrow]"}:
        return "string"
    return label


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def schema_fingerprint(frame: pd.DataFrame, target: Optional[str] = None) -> str:
    """Return a SHA-256 digest of ordered columns, dtypes, and target metadata."""
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    payload = {
        "columns": [str(column) for column in frame.columns],
        "dtypes": [_canonical_dtype(dtype) for dtype in frame.dtypes],
        "target": target,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return _digest(encoded.encode("utf-8"))


def _fingerprint_frame(frame: pd.DataFrame, mode: str, sample_rows: int) -> str:
    if mode == "schema":
        return schema_fingerprint(frame)
    if mode not in {"sampled", "full"}:
        raise ValueError("mode must be one of: schema, sampled, full")
    if not isinstance(sample_rows, int) or isinstance(sample_rows, bool) or sample_rows <= 0:
        raise ValueError("sample_rows must be a positive integer")

    selected = frame
    if mode == "sampled" and len(frame) > sample_rows:
        # Head and tail preserve deterministic boundary information without
        # pretending that a sample is a complete content representation.
        head_count = max(1, sample_rows // 2)
        selected = pd.concat([frame.head(head_count), frame.tail(sample_rows - head_count)])

    try:
        row_hashes = pd.util.hash_pandas_object(selected, index=True).to_numpy(dtype=np.uint64)
        payload = row_hashes.tobytes()
    except (TypeError, ValueError):
        # Mixed object columns can contain values that pandas cannot hash. JSON
        # is slower but deterministic and keeps the public API total.
        payload = selected.to_json(
            orient="split", date_format="iso", date_unit="ns", default_handler=_json_default
        ).encode("utf-8")
    return _digest(payload)


@dataclass(frozen=True)
class DatasetFingerprint:
    """Schema and content identity metadata for one DataFrame."""

    schema: str
    content: Optional[str]
    mode: str
    row_count: int
    feature_count: int
    target: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable fingerprint metadata."""
        return asdict(self)


def fingerprint_dataframe(
    frame: pd.DataFrame,
    mode: str = "full",
    sample_rows: int = 1000,
    target: Optional[str] = None,
) -> DatasetFingerprint:
    """Compute deterministic schema and content fingerprints.

    ``schema`` mode intentionally omits content. ``sampled`` hashes a
    deterministic head and tail sample and must not be described as a full
    dataset digest.
    """
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if mode not in {"schema", "sampled", "full"}:
        raise ValueError("mode must be one of: schema, sampled, full")
    return DatasetFingerprint(
        schema=schema_fingerprint(frame, target=target),
        content=None if mode == "schema" else _fingerprint_frame(frame, mode, sample_rows),
        mode=mode,
        row_count=int(len(frame)),
        feature_count=int(frame.shape[1] - (target in frame.columns if target else 0)),
        target=target,
    )


__all__ = ["DatasetFingerprint", "schema_fingerprint", "fingerprint_dataframe"]
