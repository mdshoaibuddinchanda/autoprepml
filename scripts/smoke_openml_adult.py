"""Run a reproducible end-to-end smoke test on OpenML's Adult dataset.

This script intentionally reports aggregate shapes and issue counts only. It
does not print rows or feature values, making it safe to use in CI logs.
"""

import argparse
import json
from pathlib import Path
from typing import Optional

import pandas as pd
from sklearn.datasets import fetch_openml

from autoprepml import AutoPrepML


def load_adult_sample(rows: Optional[int] = 10_000, seed: int = 42) -> pd.DataFrame:
    """Download Adult v2 and return a deterministic, optionally limited frame."""
    dataset = fetch_openml("adult", version=2, as_frame=True, parser="auto")
    frame = dataset.data.copy()
    frame["target"] = dataset.target
    if rows is not None:
        if rows <= 0:
            raise ValueError("rows must be positive or None")
        frame = frame.sample(n=min(rows, len(frame)), random_state=seed)
    return frame.reset_index(drop=True)


def run_smoke(rows: Optional[int] = 10_000, seed: int = 42) -> dict:
    """Run detection and classification cleaning, returning aggregate results."""
    frame = load_adult_sample(rows=rows, seed=seed)
    prep = AutoPrepML(frame, config={"reporting": {"include_plots": False}})
    detected = prep.detect(target_col="target")
    cleaned, _ = prep.clean(task="classification", target_col="target")
    return {
        "dataset": "openml:adult-v2",
        "seed": seed,
        "input_shape": list(frame.shape),
        "output_shape": list(cleaned.shape),
        "missing_columns": len(detected.get("missing_values", {})),
        "outlier_count": detected.get("outliers", {}).get("outlier_count", 0),
        "class_imbalanced": (
            bool(detected["class_imbalance"]["is_imbalanced"])
            if "class_imbalance" in detected
            else None
        ),
    }


def main(argv=None) -> int:
    """CLI entry point for the OpenML smoke test."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=10_000, help="Maximum rows to sample")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON output path (defaults to stdout)",
    )
    args = parser.parse_args(argv)
    result = run_smoke(rows=args.rows, seed=args.seed)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
