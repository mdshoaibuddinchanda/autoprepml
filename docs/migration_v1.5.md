# Migration guide for the v1.5 development API

The v1.5 development line introduces a canonical fitted workflow without
removing the existing 1.x classes. The new API is experimental until the
release gates in the v1.5 release note are complete.

## Recommended migration

Use `DataPlan` when preprocessing feeds a supervised model or a repeatable
batch inference job.

```python
from autoprepml import DataPlan

plan = DataPlan.infer(train, target="label", task="classification")
plan.fit(train)
validation_ready = plan.transform(validation)
report = plan.validate(validation)
plan.save("artifacts/data-plan.apml")
```

The plan fits imputation, scaling, and category mappings on training rows
only. `transform()` never learns from validation, test, or production rows.
Use `fit_resample()` explicitly when training data requires class balancing.

## Existing API compatibility

`AutoPrepML.clean()`, modality classes, and the chunk, storage, streaming,
tracking, and model-pipeline helpers remain available. They are useful for
exploration and established applications, but callers that need a persisted
train/test contract should migrate to `DataPlan`.

## Behaviour changes to account for

* The default `DataPlan` categorical encoding is one-hot. Use
  `{"cleaning": {"encode_method": "label"}}` only for explicitly ordinal
  features.
* Unknown categories are ignored by one-hot encoding and represented as `-1`
  by ordinal encoding. `validate()` reports these as warnings in compatible
  mode and failures in strict mode.
* A `.apml` artifact contains a checksum and a manifest. Pickle state must
  only be loaded from a trusted source.
* Text columns reported as `object` by pandas 2 and `str` by pandas 3 are
  treated as equivalent contract types and have stable schema fingerprints.

## Version and support policy

The development package version is `1.5.0.dev0`. Do not publish it as a final
release. Follow `RELEASE_CHECKLIST.md` and the v1.5 release note when the
remaining coverage, CLI, integration, benchmark, and governance gates are
complete.
