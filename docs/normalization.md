# Normalization and data processing standards

AutoPrepML treats preprocessing as part of the model contract. Every
transformation that learns a value must be fitted on training data and reused
unchanged for validation, test, and production data. The same rule applies to
imputation values, category vocabularies, feature selection, image statistics,
and text vectorizers.

This is the reason to use `TabularNormalizer.fit` followed by
`TabularNormalizer.transform`, or to put the transformer in
`make_model_pipeline`. A pipeline keeps fitting inside each training fold and
prevents validation information from leaking into a model.

## Tabular and CSV data

Read a CSV with an explicit schema whenever the source contract is known. Set
`dtype` for identifiers and categorical columns, use `parse_dates` and a
declared date format for timestamps, and list project-specific missing markers
with `na_values`. For large files, pass these same options to
`iter_chunks`; chunking changes memory use, not the schema.

The recommended order is:

1. Validate the schema, row identity, and target definition.
2. Split into train, validation, and test before fitting any statistics.
3. Impute numeric and categorical values using training-only statistics.
4. Encode categorical values with an unknown-category policy.
5. Normalize numeric features with `TabularNormalizer`.
6. Fit the estimator inside a scikit-learn pipeline.

Use `standard` scaling for features whose mean and variance are meaningful,
`robust` scaling when outliers are expected, `minmax` when a bounded range is a
model requirement, and `maxabs` for sparse or naturally zero-centred values.
Do not scale identifiers, boolean flags, target columns, or arbitrary encoded
labels. Keep the fitted normalizer with the model artifact.

```python
import pandas as pd
from sklearn.model_selection import train_test_split
from autoprepml import TabularNormalizer

frame = pd.read_csv(
    "potatoes.csv",
    dtype={"farm_id": "string", "variety": "category"},
    parse_dates=["harvested_at"],
    na_values=["", "NA", "null"],
)
train, test = train_test_split(frame, test_size=0.2, random_state=42)

normalizer = TabularNormalizer(method="robust", columns=["temperature", "yield_kg"])
train[["temperature", "yield_kg"]] = normalizer.fit_transform(
    train[["temperature", "yield_kg"]]
)
test[["temperature", "yield_kg"]] = normalizer.transform(
    test[["temperature", "yield_kg"]]
)
```

For a full mixed-type model, use `make_model_pipeline`, which performs
training-only imputation, one-hot encoding, and numeric scaling in one
serializable object.

## Image data

Convert images to a declared colour mode, resize with a declared interpolation
policy, then convert to `float32`. `zero_one` maps integer pixels from 0 through
255 to 0 through 1. `minus_one_one` maps the same values to minus 1 through 1.
For neural networks, `standard` applies per-channel `(pixel - mean) / std` on
the 0 through 1 representation. The mean and standard deviation must be
computed from the training images only and reused at inference. Augmentation
belongs in the training split and must never alter masks, bounding boxes, or
labels without a matching geometric transform.

```python
from autoprepml import ImagePrepML, fit_image_statistics

# Compute these once from training images, then store them with the model.
# train_images is an array with shape (batch, height, width, channels).
mean, std = fit_image_statistics(train_images, channel_axis=-1)

prep = ImagePrepML(
    image_dir="potato_images",
    target_size=(224, 224),
    color_mode="rgb",
    normalization_mode="standard",
    normalization_mean=mean,
    normalization_std=std,
)
prep.detect(verbose=False)
images = prep.clean(augment=False)
```

Do not use a classification image normalizer for segmentation masks or class
IDs. Preserve the original files and labels separately from processed arrays.

## Text and LLM input

Keep both the source text and the cleaned text. Normalize Unicode with NFKC by
default, remove HTML and transport artefacts according to the data contract,
and preserve meaningful punctuation and numbers unless the task explicitly
does not need them. Remove or hash personal information before sending text to
an external provider. Tokenization and vocabulary fitting must use training
text only; do not remove stopwords blindly for sentiment, negation, or legal
language tasks.

## Time series

Parse timestamps with an explicit timezone policy, sort chronologically, and
make duplicate and gap handling explicit. Use chronological splits rather than
random splits. Fit value scalers on the historical training window only. Use
`fit_normalizer(..., fit_end=n)` and then `transform_normalized()` for this
workflow. Rolling and lag features must be shifted so they do not include the
value being predicted; `add_rolling_features` defaults to that forecast-safe
behavior.

## Graph data

Keep node and edge identifiers, relation types, and topology unchanged. Apply
numeric normalization only to feature columns, using training-graph
statistics when graphs are split by time or by component. Do not normalize a
node ID or silently convert directed edges to undirected edges. Edge weights
require a domain decision before scaling.

## Chunked and parallel data

Chunking is an execution strategy, not a statistical strategy. A per-chunk
median or mean produces inconsistent results at chunk boundaries. Fit global
statistics on the training partition first, then apply the fitted transformer
to each chunk. `iter_chunks` accepts pandas schema arguments such as `dtype`,
`usecols`, `parse_dates`, and `na_values`; `iter_processed_chunks` preserves
input order and bounds pending work.

## Quality gates

Each production dataset should have a schema check, row-count and key
uniqueness check, missingness thresholds, finite numeric values, a train/test
leakage check, and a reproducible data fingerprint. Record the normalization
method, selected columns, fit-row count, library version, and source dataset
in the run report. Fail the job when these contracts are violated instead of
silently coercing data.

## References

The implementation follows scikit-learn's guidance to fit transformations on
training data and apply `transform` to later data, pandas' explicit CSV schema
and missing-value controls, and the common image convention of converting to a
float representation before applying per-channel mean and standard deviation.

* [scikit-learn common pitfalls and data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
* [scikit-learn preprocessing API](https://scikit-learn.org/stable/api/sklearn.preprocessing.html)
* [pandas `read_csv`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
* [Torchvision image transforms](https://docs.pytorch.org/vision/stable/transforms.html)
