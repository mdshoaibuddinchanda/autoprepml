# AutoPrepML v1.4.1 release candidate

AutoPrepML 1.4.1 is the patch release candidate following the published
1.4.0 release. It packages the production preprocessing safety work completed
after the 1.4.0 tag. The release is not published until the matching `v1.4.1`
tag is pushed and the trusted PyPI workflow succeeds.

## Included changes

The patch release includes the following compatibility-preserving changes:

* Train-only imputation and scaling through `fit_frame`, `TabularNormalizer`,
  and scikit-learn model pipelines.
* One-hot encoding as the high-level default for nominal categorical features;
  label encoding remains an explicit option for ordinal values.
* Target exclusion from outlier detection, removal of missing target labels,
  and a clear guard against invalid categorical SMOTE workflows.
* Configuration validation with safe defaults and fail-fast enum and numeric
  checks.
* Deterministic image ordering, SHA-256 duplicate detection, local random
  generators, and image geometry and color-mode validation.
* Chronological time-series normalization safeguards and validation for lags,
  interpolation, and resampling.
* Regression coverage for bool-only categorical pipelines, missing targets,
  and the new validation contracts.
* Documentation and repository hygiene updates. No data, generated reports,
  credentials, or failure archives are included.

## Verification gate

Before tagging the release, run the following from a clean checkout:

```bash
python -m pip install -e ".[dev,docs,notebooks]"
python -m pytest --cov=autoprepml --cov-report=term-missing --cov-fail-under=75
python scripts/validate_notebooks.py
mkdocs build --strict
python -m build
python -m twine check dist/*
```

The GitHub CI workflow must be green for Python 3.9 through 3.14, Linux,
Windows, and macOS before the `v1.4.1` tag is created. The release workflow
verifies that the tag matches the package version and publishes through the
configured PyPI trusted publisher.

## Upgrade path

Existing 1.4.0 users can install 1.4.1 without changing their public API
calls. For model training, prefer `make_model_pipeline` or a fitted
`TabularNormalizer` so preprocessing statistics are learned from training
rows only.
