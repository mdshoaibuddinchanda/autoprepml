# Creator examples

The `creator_examples/` directory contains runnable Python and notebook
workflows for maintainers who want to inspect end-to-end behavior.

The OpenML Adult v2 example:

```bash
python creator_examples/01_openml_pipeline.py --rows 300 --seed 42
```

The workflow downloads the public dataset into a temporary cache, processes it
with ordered parallel chunks, trains a scikit-learn pipeline, records metrics
in a local JSON experiment run, and deletes the temporary directory before it
returns. No data, credentials, or run artifacts are stored in Git.

Dataset reference: [UCI Adult dataset](https://archive.ics.uci.edu/dataset/2/adult).
OpenML loading details are documented in the [OpenML data guide](https://docs.openml.org/data/use/).

The notebook `creator_examples/01_openml_pipeline.ipynb` mirrors the Python
script and does not embed downloaded data.

