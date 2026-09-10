# Reproducibility

Every fitted `DataPlan` records a random seed, schema and content fingerprints,
configuration digest, package and dependency versions, and a UTC fit timestamp.
The plan's state fingerprint can be compared before and after transforming
future data to prove that inference did not mutate learned state.

`fingerprint_dataframe()` supports three modes:

* `schema` hashes ordered column names, dtypes, and target metadata.
* `sampled` hashes deterministic boundary samples and is not a complete digest.
* `full` hashes all rows and is the strongest content identity mode.

Exact repeatability also depends on deterministic dependency versions and
execution. Set `random_state` explicitly wherever sampling or balancing is
used, and record the resulting manifest with the artifact.
