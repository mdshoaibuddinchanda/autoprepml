# Leakage prevention

Any operation that learns from data must be fitted on training rows only.
This includes imputation values, scalers, category vocabularies, supervised
feature selection, learned outlier thresholds, and class balancing.

The safe order is:

1. Split raw data into training and future partitions.
2. Infer and fit a `DataPlan` on training data.
3. Transform validation and test data without fitting.
4. Resample training data only with `plan.fit_resample()` when required.
5. Validate future schemas and export the fitted artifact.

`transform()` never resamples rows. Unknown categories are handled
deterministically and reported by the contract. Outlier detection and row
removal are separate concerns: future rows should normally be flagged or
scored rather than silently deleted.

For forecasting, use chronological splits and avoid centered windows or other
operations that can read future observations unless the behavior is explicitly
requested and documented.
