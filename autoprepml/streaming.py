"""Streaming orchestration built on chunk processors and storage adapters."""

from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import pandas as pd

from .batch import DataSource, iter_processed_chunks
from .storage import StorageAdapter, get_storage_adapter


def stream_process(
    source: DataSource,
    processor: Callable[[pd.DataFrame], pd.DataFrame],
    chunksize: int = 10_000,
    n_jobs: int = 1,
    backend: str = "thread",
    adapter: Optional[StorageAdapter] = None,
    **read_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Yield processed chunks from a source without materialising the output.

    The adapter is selected from the source when omitted. Use
    :func:`write_stream` when the output should be persisted incrementally.
    """
    storage = adapter
    if storage is None and isinstance(source, (str, Path, pd.DataFrame)):
        storage = get_storage_adapter(source)

    if storage is None:
        chunks = source
    else:
        chunks = storage.iter_chunks(source, chunksize=chunksize, **read_kwargs)
        read_kwargs = {}

    yield from iter_processed_chunks(
        chunks,
        processor,
        chunksize=chunksize,
        n_jobs=n_jobs,
        backend=backend,
        **read_kwargs,
    )


def write_stream(
    source: DataSource,
    destination: Any,
    processor: Callable[[pd.DataFrame], pd.DataFrame],
    chunksize: int = 10_000,
    n_jobs: int = 1,
    backend: str = "thread",
    input_adapter: Optional[StorageAdapter] = None,
    output_adapter: Optional[StorageAdapter] = None,
    **read_kwargs: Any,
) -> Any:
    """Process and write chunks incrementally, returning the adapter result."""
    input_storage = input_adapter
    if input_storage is None and isinstance(source, (str, Path, pd.DataFrame)):
        input_storage = get_storage_adapter(source)
    output_storage = output_adapter or input_storage or get_storage_adapter(destination)
    chunks = stream_process(
        source,
        processor,
        chunksize=chunksize,
        n_jobs=n_jobs,
        backend=backend,
        adapter=input_storage,
        **read_kwargs,
    )
    return output_storage.write_chunks(chunks, destination)
