# Serialization guide

Save a fitted plan as an `.apml` artifact:

```python
from autoprepml import DataPlan

plan = DataPlan.infer(train, target="label", task="classification").fit(train)
plan.save("dataset.apml")
loaded = DataPlan.load("dataset.apml")
ready = loaded.transform(test)
```

Writes are atomic. The archive contains a JSON manifest and a checksummed
serialized fitted state. Corruption raises `ArtifactError` before state is
used, and incompatible artifact formats are rejected.

Python serialization can execute arbitrary code during loading. Never load an
`.apml` file from an untrusted source. A checksum detects accidental
corruption; it is not a security boundary.
