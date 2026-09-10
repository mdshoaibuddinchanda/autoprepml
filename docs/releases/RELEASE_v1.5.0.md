# AutoPrepML v1.5.0

AutoPrepML 1.5.0 is the production release of the fitted data-readiness
workflow. It keeps the existing modality APIs while adding explicit contracts,
train-only fitting, reproducible identity, lineage, and artifact boundaries.

## Release highlights

* `DataPlan` provides `infer`, `fit`, `transform`, `validate`, `fit_resample`,
  `save`, and `load` lifecycle operations.
* `DataContract` and structured validation reports identify missing, unexpected,
  incompatible, null, range, category, order, and uniqueness issues.
* Schema and content fingerprints support schema-only, sampled, and full modes
  without placing raw records in reports.
* Checksummed `.apml` artifacts contain versioned manifests, transformation
  lineage, dependency metadata, and reproducibility state.
* Chunked and ordered parallel execution, storage adapters, streaming output,
  local experiment tracking, MLflow integration, and sklearn model pipelines
  support larger production workflows.
* Forecast-safe time-series features, train-only normalization, image pixel
  conventions, and validated structured LLM responses reduce leakage and
  integration risk.
* The CLI supports inspect, fit, validate, transform, and version operations
  with JSON output suitable for automation.

## Quality gates

The release was validated by the complete test suite and the GitHub Actions
matrix on Python 3.9 through 3.14, Windows, macOS, and Linux. The enforced gate
is 90 percent branch-aware coverage; the local release baseline is 513 passed,
3 skipped, and 90.32 percent coverage. CI also runs Ruff, Black, strict typing
for the v1.5 core, Bandit, pip-audit, documentation builds, notebook execution,
and a clean wheel and source distribution smoke test.

## Data and examples

The maintained notebooks use deterministic synthetic data. The creator workflow
can download the public OpenML Adult dataset into a temporary directory for a
smoke run, and removes the dataset and generated artifacts before completing.
No datasets, model files, credentials, or generated reports are committed.

## Compatibility and trust

The package requires Python 3.9 or newer. Existing 1.x modality classes remain
available. Unknown categories are handled deterministically by the fitted
pipeline, and pandas 2 and pandas 3 text dtype labels are treated as equivalent
at the contract boundary. `.apml` state uses pickle internally; load artifacts
only from trusted sources, because a checksum detects corruption but cannot make
an untrusted pickle safe.

## Follow-up roadmap

Future work includes broader live-provider and object-store integration tests,
more benchmark history, and a carefully reviewed 100 percent branch-coverage
objective. These improvements are additive and do not lower the 90 percent
release gate.

See the [migration guide](../migration_v1.5.md), [API reference](../api_reference.md),
and [root README](https://github.com/mdshoaibuddinchanda/autoprepml/blob/main/README.md)
for usage details.
