# Limitations and boundaries

AutoPrepML prepares and validates data. It does not select models, train
neural networks, optimize hyperparameters, serve models, or provide a hosted
observability service.

`DataPlan` currently targets tabular fitted preprocessing. Text, image,
time-series, and graph classes retain their established deterministic APIs;
their modality-specific deviations are documented in the existing modality
guides.

`.apml` artifacts contain Python serialized estimator state. Loading a pickle
from an untrusted source can execute arbitrary code. Checksums detect
corruption but do not make an untrusted artifact safe. Only load artifacts
from a trusted producer.

Chunked and streamed callbacks must not fit statistical transformers
independently per chunk. Fit once on training data, then apply the fitted
transformer to each bounded chunk.
