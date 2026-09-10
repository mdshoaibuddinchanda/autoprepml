"""Measure the v1.5 fitted and chunked execution paths on synthetic data.

The benchmark is intentionally self-contained. It writes its temporary CSV
and all intermediate artifacts below a temporary directory, then emits only
aggregate timing and throughput metrics as JSON.
"""

from __future__ import annotations

import argparse
import json
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from autoprepml import DataPlan, process_chunks


def _identity_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Benchmark a row-preserving chunk callback."""
    return chunk


def _package_version() -> str:
    try:
        return version("autoprepml")
    except PackageNotFoundError:
        return "source-tree"


def make_frame(rows: int, seed: int) -> pd.DataFrame:
    """Create deterministic numeric data without persisting it."""
    if not isinstance(rows, int) or isinstance(rows, bool) or rows < 1:
        raise ValueError("rows must be a positive integer")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise TypeError("seed must be an integer")
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "feature_a": generator.normal(size=rows),
            "feature_b": generator.uniform(size=rows),
            "feature_c": generator.integers(0, 100, size=rows),
            "target": generator.integers(0, 2, size=rows),
        }
    )


def run_benchmark(
    rows: int = 20_000,
    chunksize: int = 1_000,
    n_jobs: int = 1,
    seed: int = 42,
) -> Dict[str, Any]:
    """Run fitted-plan and chunked throughput measurements."""
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer")

    frame = make_frame(rows, seed)
    plan_start = perf_counter()
    plan = DataPlan.infer(frame, target="target", task="classification").fit(frame)
    transformed = plan.transform(frame)
    plan_seconds = perf_counter() - plan_start

    with TemporaryDirectory(prefix="autoprepml-benchmark-") as temporary:
        source = Path(temporary) / "synthetic.csv"
        frame.to_csv(source, index=False)
        chunk_start = perf_counter()
        processed = process_chunks(
            source,
            _identity_chunk,
            chunksize=chunksize,
            n_jobs=n_jobs,
        )
        chunk_seconds = perf_counter() - chunk_start

    return {
        "package_version": _package_version(),
        "rows": rows,
        "columns": frame.shape[1],
        "chunksize": chunksize,
        "n_jobs": n_jobs,
        "seed": seed,
        "plan_transform_seconds": round(plan_seconds, 6),
        "plan_rows_per_second": round(rows / max(plan_seconds, 1e-12), 2),
        "chunk_seconds": round(chunk_seconds, 6),
        "chunk_rows_per_second": round(len(processed) / max(chunk_seconds, 1e-12), 2),
        "transformed_shape": list(getattr(transformed, "shape", ())),
        "chunk_shape": list(processed.shape),
    }


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=20_000)
    parser.add_argument("--chunksize", type=int, default=1_000)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = run_benchmark(
        rows=args.rows,
        chunksize=args.chunksize,
        n_jobs=args.n_jobs,
        seed=args.seed,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
