# Transformation lineage

`DataPlan.manifest()` returns machine-readable provenance for a fitted plan.
It includes input and output columns, transformations, contract and dataset
fingerprints, configuration digest, dependency versions, random seed, and UTC
fit time. It contains no raw records, prompts, or credentials.

```python
manifest = plan.manifest()
print(manifest["transformations"])
```

The manifest is embedded in `.apml` artifacts alongside the fitted
transformation state. Use it to compare runs, review schema changes, and
explain how a model input was produced.
