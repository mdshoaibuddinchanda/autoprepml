# DataPlan architecture

`DataPlan` is the recommended v1.5 workflow for ML data readiness. It records
the expected feature schema, learns transformations from training rows, and
produces deterministic feature matrices for later partitions.

```text
UNFITTED
  fit(training)
FITTED
  transform(validation, test, production)
  validate(future_data)
  save(dataset.apml)
```

The plan does not train a model. It prepares data and can be composed with
scikit-learn estimators through `make_model_pipeline`.

For a compact operational summary, call `plan.readiness_report(future_data)`.
The report separates schema warnings from blocking failures and records
evidence for leakage safety, reproducibility, dataset identity, and lineage.

## State rules

Calling `transform()` before `fit()` raises `NotFittedError`. Fitting never
mutates the caller's DataFrame. Transforming future data never changes fitted
statistics, category mappings, output columns, or the state fingerprint.

## Feature order and categories

The contract captures the training feature order. Extra columns are reported
and ignored in compatible mode. Missing required features are blocking
errors. One-hot encoding uses deterministic unknown-category handling; an
unknown category produces a validation warning and an all-zero one-hot block.

## Whole-dataset cleaning versus fitted preprocessing

`AutoPrepML.clean()` remains useful for data-quality exploration where a single
dataset is being cleaned as a whole. It must not be used as a replacement for
train-only fitting in a supervised evaluation. `DataPlan` is the boundary that
makes this distinction explicit.
