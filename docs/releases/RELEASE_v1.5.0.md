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
* Documentation for architecture, leakage prevention, contracts,
  reproducibility, lineage, serialization, and product limitations.

## Remaining release gates

The 1.5.0 release is not ready to publish. Remaining work includes storage and
tracking coverage, time-series and supervised feature-engineering audits,
structured LLM suggestions, CLI subcommands, strict static typing, property
tests, branch-coverage improvement, reproducible benchmarks, research
experiments, governance files, and a migration guide.

The current development branch must continue to pass the existing Python
3.9–3.14 CI matrix and must not lower the current coverage gate. The final
release requires a fresh-environment wheel smoke test and a complete audit of
all documented examples.
