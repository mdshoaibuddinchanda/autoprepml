# Migration guide for the v1.5 API

The v1.5 release introduces a canonical fitted workflow without removing the
existing 1.x classes. Use the new API when preprocessing feeds a supervised
model or a repeatable batch inference job.

## Recommended migration

Use `DataPlan` when preprocessing feeds a supervised model or a repeatable
batch inference job.

<!-- executable -->
```python
from pathlib import Path
import tempfile

import pandas as pd
from autoprepml import DataPlan

train = pd.DataFrame(
    {
        "age": [20, 30, 40, 50],
        "country": ["GB", "US", "GB", "US"],
        "label": [0, 1, 0, 1],
    }
)
validation = pd.DataFrame({"age": [35], "country": ["FR"]})
plan = DataPlan.infer(train, target="label", task="classification")
plan.fit(train)
validation_ready = plan.transform(validation)
report = plan.validate(validation)
with tempfile.TemporaryDirectory() as directory:
    plan.save(Path(directory) / "data-plan.apml")
assert validation_ready.shape == (1, 3)
assert report.status == "WARN"
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

The package version is `1.5.0`. Follow `RELEASE_CHECKLIST.md` for subsequent
patch releases and keep artifact loading restricted to trusted sources.
