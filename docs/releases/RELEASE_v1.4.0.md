# AutoPrepML v1.4.0

AutoPrepML 1.4.0 adds bounded large-data execution and integrations for
storage, streaming, experiment tracking, and scikit-learn model pipelines.

## Highlights

### Chunked and parallel execution

`iter_chunks`, `iter_processed_chunks`, and `process_chunks` support DataFrame,
CSV, JSON, JSONL, and Parquet inputs. Thread execution is the default, process
execution is opt in, and queued work is bounded while output order remains
deterministic.

### Storage and streaming

`LocalStorageAdapter` provides atomic local writes, while
`InMemoryStorageAdapter` supports tests and notebooks. `FsspecStorageAdapter`
is available for optional remote filesystems. `stream_process` and
`write_stream` connect these adapters to chunk processors without materialising
the entire result.

### Experiment and model integrations

`LocalExperimentTracker` records JSON manifests, parameters, metrics, and
artifacts. `MLflowExperimentTracker` is optional and lazy. The new
`make_preprocessing_pipeline` and `make_model_pipeline` helpers create
scikit-learn pipelines that fit preprocessing state on training data only.

### Creator examples

`creator_examples/01_openml_pipeline.py` and its notebook demonstrate the full
workflow against OpenML Adult v2. The dataset and experiment artifacts are
created inside a temporary directory and deleted automatically; no data is
checked into the repository.

## Compatibility

Python 3.9 through 3.14 remain supported. The existing modality APIs are
unchanged. Install the optional `fsspec` or `mlflow` dependencies only when
those integrations are needed.

