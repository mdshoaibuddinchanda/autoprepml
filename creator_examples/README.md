# Creator examples

These examples are runnable demonstrations for maintainers and library
creators. They download a versioned public dataset into a temporary directory,
run AutoPrepML, record an experiment manifest, and remove all generated files
when the process exits.

The examples never write datasets, credentials, model artifacts, or experiment
runs into the repository. Run them from an environment with network access:

```bash
python creator_examples/01_openml_pipeline.py --rows 300
```

The dataset is OpenML Adult, version 2, a public mirror of the UCI Adult
income dataset. The script uses only aggregate output in the terminal. The
dataset license and source are documented by [UCI](https://archive.ics.uci.edu/dataset/2/adult)
and the [OpenML data guide](https://docs.openml.org/data/use/).

The matching notebook is `01_openml_pipeline.ipynb`. It contains the same
workflow as executable cells and does not embed downloaded data.

