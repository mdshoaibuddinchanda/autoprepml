# Command line reference

The v1.5 command family operates on a fitted `DataPlan`. The original
flag-based command remains available for exploratory whole-dataset cleaning.

## Inspect

Inspect a supported local CSV, JSON, JSONL, or Parquet file without fitting a
transformation:

```bash
autoprepml inspect train.csv --target label --task classification
```

The command prints a JSON schema, feature list, and deterministic fingerprint.

## Fit

Fit transformations on training data and save a reusable artifact:

```bash
autoprepml fit train.csv --target label --task classification --output plan.apml
```

Use `--fingerprint-mode schema` for schema-only provenance or
`--fingerprint-mode sampled --fingerprint-sample-rows 1000` for bounded content
diagnostics. Use `--output-format sparse` when one-hot output is large.

## Validate

Validate future data against the saved contract. Compatible mode reports
recoverable drift as warnings; strict mode promotes drift to failures:

```bash
autoprepml validate validation.csv --plan plan.apml
autoprepml validate validation.csv --plan plan.apml --strict
```

The command returns status zero for compatible data and status one when strict
validation finds a blocking issue.

## Transform

Transform data without fitting again:

```bash
autoprepml transform validation.csv --plan plan.apml --output validation-ready.csv
```

The plan's output format must be `pandas` or `sparse` for file output. A plan
configured for NumPy output is intended for Python callers.

## Version

```bash
autoprepml version
```

The legacy interface remains available for compatibility:

```bash
autoprepml --input data.csv --output cleaned.csv --report report.json
```
