# Examples

AutoPrepML's maintained examples are deterministic Jupyter notebooks. They
are designed to be read from top to bottom and run without private data.

## Notebook catalogue

| Notebook | Coverage |
| --- | --- |
| `01_tabular_quality.ipynb` | Detection, cleaning, EDA, feature engineering, and a leakage-safe model pipeline |
| `02_text_nlp.ipynb` | Text quality checks, normalization, tokenization, and feature extraction |
| `03_time_series.ipynb` | Timestamp validation, gap filling, interpolation, lags, and rolling features |
| `04_graph_data.ipynb` | Node and edge integrity, degree features, components, and adjacency output |
| `05_image_data.ipynb` | Temporary synthetic image creation, validation, resizing, normalization, and augmentation |
| `06_scalable_pipeline.ipynb` | Chunked processing, bounded parallelism, streaming, storage, and experiment tracking |
| `07_llm_integration.ipynb` | Provider selection, safe configuration, and an explicitly opt-in request |
| `08_normalization_standards.ipynb` | Train-only normalization for tabular, image, text, and time-series data |

The notebooks live in `examples/notebooks/` in the source repository. Install
the notebook extra before running them:

```bash
python -m pip install -e ".[dev,notebooks]"
python scripts/validate_notebooks.py
```

The validator runs each notebook in a temporary working directory and stops on
the first error. The image example creates files only in a temporary directory
and removes them before completion. The LLM example makes no network request
unless `AUTOPREPML_RUN_LLM_EXAMPLE=1` is set explicitly.

## Public-data creator workflow

The separate `creator_examples/` directory contains a workflow based on the
versioned OpenML Adult v2 dataset. It downloads the data into a temporary cache,
processes it in ordered parallel chunks, records a local experiment manifest,
and removes all generated files. See the [creator workflow guide](creator_examples.md).
