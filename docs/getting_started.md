# Getting started

AutoPrepML has two compatible workflows.

Use `AutoPrepML.clean()` for exploratory or generic whole-dataset cleaning.
Use `DataPlan` when a supervised model will consume train, validation, test,
or production data. A `DataPlan` learns preprocessing statistics once and
reuses them without fitting on future rows.

## Install

```bash
python -m pip install autoprepml
```

Install development and notebook tools from a checkout with:

```bash
python -m pip install -e ".[dev,docs,notebooks]"
```

## Canonical fitted workflow

```python
import pandas as pd
from autoprepml import DataPlan

train = pd.read_csv("train.csv")
valid = pd.read_csv("valid.csv")

plan = DataPlan.infer(train, target="label", task="classification")
plan.fit(train)
valid_ready = plan.transform(valid)
report = plan.validate(valid)
plan.save("dataset.apml")
```

`transform()` returns predictors in the fitted feature order. Unknown
categories are ignored by one-hot encoding and reported as warnings. Missing
required columns or duplicate columns are blocking contract errors.

## Existing cleaning API

```python
from autoprepml import AutoPrepML

cleaned, report = AutoPrepML(frame).clean()
```

This API is intentionally retained for backwards compatibility. For model
training, split first and prefer `DataPlan.fit()` and `DataPlan.transform()`.
