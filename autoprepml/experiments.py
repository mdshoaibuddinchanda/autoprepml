"""Local and optional MLflow experiment tracking integrations."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Optional
from uuid import uuid4


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentRun:
    """A local JSON-backed experiment run."""

    tracker: "LocalExperimentTracker"
    run_id: str
    name: str
    tags: Dict[str, str] = field(default_factory=dict)
    started_at: str = field(default_factory=_utc_now)
    status: str = "running"
    params: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, str] = field(default_factory=dict)

    def __enter__(self) -> "ExperimentRun":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.end(status="failed" if exc_type else "finished")

    def log_params(self, params: Dict[str, Any], **kwargs: Any) -> None:
        """Record JSON-serializable run parameters."""
        del kwargs
        if not isinstance(params, dict):
            raise TypeError("params must be a dictionary")
        self.params.update(params)
        self.tracker._write_run(self)

    def log_metrics(self, metrics: Dict[str, float], **kwargs: Any) -> None:
        """Record numeric run metrics."""
        del kwargs
        if not isinstance(metrics, dict):
            raise TypeError("metrics must be a dictionary")
        for name, value in metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"metric '{name}' must be numeric")
            self.metrics[str(name)] = float(value)
        self.tracker._write_run(self)

    def log_artifact(self, path: Any, artifact_name: Optional[str] = None) -> Path:
        """Copy an artifact into the run directory and record its path."""
        source = Path(path)
        if not source.is_file():
            raise FileNotFoundError(f"Artifact does not exist: {source}")
        destination_name = artifact_name or source.name
        if Path(destination_name).name != destination_name:
            raise ValueError("artifact_name must be a file name, not a path")
        destination = self.tracker._run_dir(self.run_id) / "artifacts" / destination_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        self.artifacts[destination_name] = str(destination)
        self.tracker._write_run(self)
        return destination

    def log_plan(self, plan: Any, artifact_name: str = "data-plan.json") -> Path:
        """Persist a plan manifest as a non-tabular lineage artifact."""
        if not hasattr(plan, "manifest") or not callable(plan.manifest):
            raise TypeError("plan must provide a manifest() method")
        if Path(artifact_name).name != artifact_name or not artifact_name.endswith(".json"):
            raise ValueError("artifact_name must be a JSON file name")
        manifest = plan.manifest()
        if not isinstance(manifest, dict):
            raise TypeError("plan.manifest() must return a dictionary")
        destination = self.tracker._run_dir(self.run_id) / "artifacts" / artifact_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        try:
            temporary.write_text(
                json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            temporary.replace(destination)
        finally:
            if temporary.exists():
                temporary.unlink()
        self.artifacts[artifact_name] = str(destination)
        self.tracker._write_run(self)
        return destination

    def end(self, status: str = "finished") -> None:
        """Finalize the run and persist its manifest."""
        if status not in {"finished", "failed", "cancelled"}:
            raise ValueError("status must be finished, failed, or cancelled")
        self.status = status
        self.tracker._write_run(self)


class LocalExperimentTracker:
    """Dependency-free experiment tracker storing manifests as JSON files."""

    def __init__(self, root_dir: Any = "runs"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _run_dir(self, run_id: str) -> Path:
        return self.root_dir / run_id

    def _write_run(self, run: ExperimentRun) -> None:
        run_dir = self._run_dir(run.run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "run_id": run.run_id,
            "name": run.name,
            "tags": run.tags,
            "started_at": run.started_at,
            "updated_at": _utc_now(),
            "status": run.status,
            "params": run.params,
            "metrics": run.metrics,
            "artifacts": run.artifacts,
        }
        manifest = run_dir / "run.json"
        with NamedTemporaryFile(
            mode="w", encoding="utf-8", prefix=".run-", suffix=".tmp", dir=run_dir, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(payload, temporary, indent=2, default=str)
            temporary.write("\n")
        try:
            temporary_path.replace(manifest)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def start_run(
        self,
        name: str = "autoprepml-run",
        tags: Optional[Dict[str, str]] = None,
    ) -> ExperimentRun:
        """Create a run; use it as a context manager for guaranteed finalization."""
        if not name or not isinstance(name, str):
            raise ValueError("name must be a non-empty string")
        run = ExperimentRun(
            tracker=self,
            run_id=uuid4().hex,
            name=name,
            tags={str(key): str(value) for key, value in (tags or {}).items()},
        )
        self._write_run(run)
        return run

    def list_runs(self):
        """Return persisted run manifests ordered by newest modification."""
        manifests = list(self.root_dir.glob("*/run.json"))
        return [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(manifests, key=lambda item: item.stat().st_mtime, reverse=True)
        ]


class MLflowExperimentTracker:
    """Optional MLflow adapter loaded only when explicitly instantiated."""

    def __init__(self, experiment_name: str = "AutoPrepML", tracking_uri: Optional[str] = None):
        try:
            import mlflow
        except ImportError as error:
            raise ImportError(
                "MLflowExperimentTracker requires the optional 'mlflow' package"
            ) from error
        self._mlflow = mlflow
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, name: str = "autoprepml-run", tags: Optional[Dict[str, str]] = None):
        """Return an MLflow run context with the same logging method names."""
        tracker = self._mlflow

        class _RunContext:
            def __enter__(self_inner):
                self_inner._context = tracker.start_run(run_name=name, tags=tags or {})
                self_inner._run = self_inner._context.__enter__()
                return self_inner

            def __exit__(self_inner, exc_type, exc_value, traceback):
                return self_inner._context.__exit__(exc_type, exc_value, traceback)

            def log_params(self_inner, params, **kwargs):
                del kwargs
                tracker.log_params(params)

            def log_metrics(self_inner, metrics, **kwargs):
                del kwargs
                tracker.log_metrics(metrics)

            def log_artifact(self_inner, path, artifact_name=None):
                del artifact_name
                tracker.log_artifact(str(path))

            def log_plan(self_inner, plan, artifact_name="data-plan.json"):
                if not hasattr(plan, "manifest") or not callable(plan.manifest):
                    raise TypeError("plan must provide a manifest() method")
                manifest = plan.manifest()
                temporary = NamedTemporaryFile(
                    mode="w", encoding="utf-8", suffix=".json", delete=False
                )
                temporary_path = Path(temporary.name)
                try:
                    json.dump(manifest, temporary, indent=2, sort_keys=True)
                    temporary.write("\n")
                    temporary.close()
                    tracker.log_artifact(str(temporary_path), artifact_path=artifact_name)
                finally:
                    temporary_path.unlink(missing_ok=True)

        return _RunContext()
