# Data contracts

`DataContract` is a lightweight compatibility boundary. It captures expected
column names, dtypes, semantic roles, nullability, categories, bounds, and
optional uniqueness constraints without attempting to replace a full data
observability platform.

Validation supports two modes:

* `compatible` reports unexpected columns, dtype changes, and unseen categories
  as warnings while still allowing safe transformations.
* `strict` promotes those compatibility changes to failures.

Results are structured `ValidationReport` and `ValidationIssue` objects. Every
issue includes a code, severity, column, message, expected value, and observed
value. Reports can be exported with `to_dict()` for JSON APIs.

```python
report = plan.validate(future_frame, mode="compatible")
if not report.compatible:
    report.raise_for_error()
```

Pandera export is optional. Install Pandera separately and call
`plan.contract.to_pandera()` when a downstream validation system requires it.
