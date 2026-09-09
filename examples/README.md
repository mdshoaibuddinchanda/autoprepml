# AutoPrepML examples

This directory contains the maintained examples for AutoPrepML. The examples
are intentionally small, deterministic, and runnable from a clean checkout.
They use synthetic data unless a creator workflow explicitly names a public
dataset.

## Notebook index

| Notebook | Scope | External services |
| --- | --- | --- |
| `notebooks/01_tabular_quality.ipynb` | Detection, cleaning, EDA, feature engineering, and a leakage-safe model pipeline | None |
| `notebooks/02_text_nlp.ipynb` | Text quality checks, cleaning, tokenization, and text features | None |
| `notebooks/03_time_series.ipynb` | Timestamp validation, gap filling, lags, and forecast-safe rolling features | None |
| `notebooks/04_graph_data.ipynb` | Node and edge validation, graph features, and components | None |
| `notebooks/05_image_data.ipynb` | Synthetic image creation, validation, resizing, normalization, and augmentation | Pillow only |
| `notebooks/06_scalable_pipeline.ipynb` | Chunking, bounded parallel processing, streaming, storage adapters, and experiment tracking | None |
| `notebooks/07_llm_integration.ipynb` | Provider configuration and safe optional LLM execution | Optional provider and credentials |

The notebook cells display bounded tabular or textual outputs. They do not
embed images or commit generated data. The image example writes files only
inside a temporary directory and removes that directory before it finishes.

## Run the examples

Install the notebook tooling and the library in an isolated environment:

```bash
python -m pip install -e ".[dev,notebooks]"
```

Execute one notebook from the repository root:

```bash
python -m jupyter nbconvert --execute --to notebook --inplace \
  examples/notebooks/01_tabular_quality.ipynb
```

Execute the full maintained notebook suite:

```bash
python scripts/validate_notebooks.py
```

The validator executes notebooks in temporary working directories and fails
on the first cell error. It does not write notebook outputs back to the
repository unless `--inplace` is explicitly requested.

## Creator workflow

`creator_examples/` is separate from the deterministic examples. Its OpenML
workflow downloads the versioned Adult v2 dataset into a temporary cache,
records a local experiment manifest, and removes all temporary artifacts.
See `creator_examples/README.md` for the data source and reproducibility notes.
