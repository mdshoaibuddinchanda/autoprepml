"""Creator workflow: OpenML data, chunk processing, tracking, and modeling.

All downloaded data and run artifacts live inside a TemporaryDirectory and
are deleted automatically. No dataset or credential is written to the repo.
"""

import argparse
import json
from pathlib import Path
import tempfile
from typing import Any, Dict

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from autoprepml import (
    AutoPrepML,
    LocalExperimentTracker,
    make_model_pipeline,
    process_chunks,
)


def clean_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """Apply the library's current stateless cleaning API to one chunk."""
    prep = AutoPrepML(chunk, config={"reporting": {"include_plots": False}})
    cleaned, _ = prep.clean()
    return cleaned


def run_creator_example(rows: int = 300, seed: int = 42) -> Dict[str, Any]:
    """Run the complete creator workflow and return aggregate metrics."""
    if rows < 20:
        raise ValueError("rows must be at least 20 for a train/test example")

    with tempfile.TemporaryDirectory(prefix="autoprepml-creator-") as temporary:
        work_dir = Path(temporary)
        dataset_home = work_dir / "dataset-cache"
        dataset = fetch_openml(
            "adult",
            version=2,
            as_frame=True,
            parser="auto",
            data_home=str(dataset_home),
        )
        frame = dataset.data.copy()
        frame["target"] = dataset.target
        frame = frame.sample(n=min(rows, len(frame)), random_state=seed).reset_index(drop=True)

        source_path = work_dir / "adult.csv"
        frame.to_csv(source_path, index=False)
        cleaned = process_chunks(source_path, clean_chunk, chunksize=100, n_jobs=2)

        features = frame.drop(columns="target")
        target = frame["target"]
        x_train, x_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=seed, stratify=target
        )
        pipeline = make_model_pipeline(
            frame,
            LogisticRegression(max_iter=300),
            target_col="target",
        )
        pipeline.fit(x_train, y_train)
        accuracy = accuracy_score(y_test, pipeline.predict(x_test))

        cleaned_path = work_dir / "cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)
        tracker = LocalExperimentTracker(work_dir / "runs")
        with tracker.start_run(
            "openml-adult-creator",
            tags={"dataset": "openml:adult-v2", "temporary_data": "true"},
        ) as run:
            run.log_params({"rows": len(frame), "chunksize": 100, "n_jobs": 2, "seed": seed})
            run.log_metrics({"accuracy": accuracy, "cleaned_rows": len(cleaned)})
            run.log_artifact(cleaned_path)

        return {
            "dataset": "openml:adult-v2",
            "seed": seed,
            "input_shape": list(frame.shape),
            "chunk_cleaned_shape": list(cleaned.shape),
            "accuracy": round(float(accuracy), 6),
            "temporary_artifacts_deleted": True,
        }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    print(json.dumps(run_creator_example(rows=args.rows, seed=args.seed), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
