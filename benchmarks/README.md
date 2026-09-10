# Reproducible benchmarks

The v1.5 benchmark harness measures the fitted `DataPlan` path and ordered
chunk processing on deterministic synthetic data. It does not download data,
write outputs into the repository, or claim a universal performance number.
Run it from a development environment with:

```bash
python scripts/benchmark_v15.py --rows 20000 --chunksize 1000 --n-jobs 2
```

Use the same Python version, dependency lock, row count, chunk size, worker
count, and seed when comparing commits. Record the JSON output externally with
the machine and operating-system details. Benchmark results are evidence for
an environment, not a release gate by themselves.
