# AutoPrepML v1.5 development plan

The branch is versioned `1.5.0.dev0`; no v1.5.0 package or PyPI release has
been published.

Version 1.5 is an engineering consolidation release. It makes ML data
readiness the centre of the project: validated schemas, train-only fitting,
reproducible fingerprints, transformation lineage, and reusable artifacts.
It does not add another modality or become a model-training framework.

## Delivered in the current development branch

* `DataPlan` lifecycle with explicit `infer`, `fit`, `transform`, `validate`,
  `fit_resample`, `save`, and `load` operations.
* Lightweight `DataContract`, `ColumnContract`, `ValidationIssue`, and
  `ValidationReport` objects.
* Deterministic schema and content fingerprints with schema, sampled, and full
  modes.
* Checksummed, atomic `.apml` artifacts with machine-readable manifests.
* Stable exception hierarchy and a shared `PrepProtocol` typing boundary.
* Structured readiness reports covering schema integrity, leakage safety,
  reproducibility, dataset identity, and lineage.
* Modern CLI subcommands for `inspect`, `fit`, `validate`, `transform`, and
  `version`, with JSON output suitable for automation.
* Bounded sparse and dense transform policies, with a configurable guard
  against accidental sparse-to-dense memory explosions.
* Hardened local, in-memory, and fsspec storage paths with format validation,
  empty-input handling, atomic replacement, and file synchronization.
* Experiment protocols and plan-manifest logging for local and optional MLflow
  runs, plus a train-only fitted feature selector.
* Executable canonical documentation examples, property tests, and strict
  static typing for the v1.5 core modules.
* Forecast-safe time-series lag and rolling features that require chronological
  input, and historical-only normalization boundaries.
* Validated structured LLM analysis and feature recommendations. Model output
  remains advisory and is never executed as configuration.
* Documentation for architecture, leakage prevention, contracts,
  reproducibility, lineage, serialization, and product limitations.
* Compatibility with pandas 2 text columns reported as `object` and pandas 3
  text columns reported as `str`, including stable schema fingerprints.

## Remaining release gates

The 1.5.0 release is not ready to publish. Remaining work is concentrated on
raising and enforcing coverage for every public branch, adding reproducible
benchmark and research workflows, completing broader adapter and optional
provider integration tests, and finishing final release hardening. Governance
files and the migration guide are included in the development branch.

The current development branch must continue to pass the Python 3.9-3.14 CI
matrix and must not lower the current coverage gate. The final release also
requires a fresh-environment wheel smoke test, complete documentation and
notebook execution, attached GitHub Release artifacts, and an explicit review
of branch protection and trusted publishing settings.
