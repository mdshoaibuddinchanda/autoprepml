"""Chunked and parallel tabular processing helpers.

The helpers in this module keep input reads bounded and preserve chunk order
when work is parallelised. They are deliberately processor-agnostic: callers
provide a function that accepts one :class:`pandas.DataFrame` and returns a
DataFrame. This makes the execution layer usable with AutoPrepML, custom
transformers, and future streaming backends.
"""

from collections import deque
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

import pandas as pd


DataSource = Union[str, Path, pd.DataFrame, Iterator[pd.DataFrame]]
ChunkProcessor = Callable[[pd.DataFrame], pd.DataFrame]


def _validate_execution_options(chunksize: int, n_jobs: int, backend: str) -> None:
    """Validate shared chunk execution options."""
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")
    if not isinstance(n_jobs, int) or isinstance(n_jobs, bool) or n_jobs == 0:
        raise ValueError("n_jobs must be a non-zero integer")
    if backend not in {"thread", "process"}:
        raise ValueError("backend must be 'thread' or 'process'")


def _yield_frame_chunks(frame: pd.DataFrame, chunksize: int) -> Iterator[pd.DataFrame]:
    """Yield bounded copies from an in-memory frame."""
    for start in range(0, len(frame), chunksize):
        yield frame.iloc[start : start + chunksize].copy()


def iter_chunks(
    source: DataSource,
    chunksize: int = 10_000,
    **read_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Yield DataFrame chunks from a frame, path, or DataFrame iterator.

    CSV input uses pandas' native streaming reader. JSON and Parquet input are
    loaded once and then partitioned because pandas does not expose a portable
    chunk reader for those formats. The function never writes to ``source``.

    Args:
        source: DataFrame, local path, or iterable yielding DataFrames.
        chunksize: Maximum number of rows yielded per chunk.
        read_kwargs: Additional keyword arguments passed to the pandas reader.
    """
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")

    if isinstance(source, pd.DataFrame):
        yield from _yield_frame_chunks(source, chunksize)
        return

    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input data source does not exist: {path}")
        suffixes = "".join(path.suffixes).lower()
        if suffixes.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz")):
            options = dict(read_kwargs)
            options.pop("chunksize", None)
            reader = pd.read_csv(path, chunksize=chunksize, **options)
            yield from reader
            return
        if suffixes.endswith((".json", ".jsonl")):
            frame = pd.read_json(path, **read_kwargs)
            yield from _yield_frame_chunks(frame, chunksize)
            return
        if suffixes.endswith(".parquet"):
            frame = pd.read_parquet(path, **read_kwargs)
            yield from _yield_frame_chunks(frame, chunksize)
            return
        raise ValueError("Unsupported input format; use CSV, JSON, JSONL, or Parquet")

    if hasattr(source, "__iter__"):
        for chunk in source:
            if not isinstance(chunk, pd.DataFrame):
                raise TypeError("DataFrame iterators must yield pandas DataFrame objects")
            yield from _yield_frame_chunks(chunk, chunksize)
        return

    raise TypeError("source must be a DataFrame, path, or iterable of DataFrames")


def _validate_result(result: Any) -> pd.DataFrame:
    """Ensure a processor returned a DataFrame with a useful error."""
    if not isinstance(result, pd.DataFrame):
        raise TypeError("chunk processor must return a pandas DataFrame")
    return result


def _parallel_results(
    chunks: Iterator[pd.DataFrame],
    processor: ChunkProcessor,
    executor: Executor,
    max_pending: int,
) -> Iterator[pd.DataFrame]:
    """Submit a bounded number of futures while preserving input order."""
    pending = deque()
    exhausted = False

    while not exhausted or pending:
        while not exhausted and len(pending) < max_pending:
            try:
                chunk = next(chunks)
            except StopIteration:
                exhausted = True
                break
            pending.append(executor.submit(processor, chunk))

        if pending:
            yield _validate_result(pending.popleft().result())


def iter_processed_chunks(
    source: DataSource,
    processor: ChunkProcessor,
    chunksize: int = 10_000,
    n_jobs: int = 1,
    backend: str = "thread",
    max_pending: Optional[int] = None,
    **read_kwargs: Any,
) -> Iterator[pd.DataFrame]:
    """Process input chunks and yield results in deterministic input order.

    ``backend='thread'`` is the safest default for pandas workloads and custom
    callables. ``backend='process'`` can improve CPU-bound workloads, but the
    processor must be importable and pickleable (especially on Windows).
    ``max_pending`` limits queued work to avoid loading an entire input into
    memory before results are consumed.
    """
    _validate_execution_options(chunksize, n_jobs, backend)
    if not callable(processor):
        raise TypeError("processor must be callable")

    chunks = iter_chunks(source, chunksize=chunksize, **read_kwargs)
    workers = n_jobs if n_jobs > 0 else 1
    if n_jobs < 0:
        import os

        workers = max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    if workers == 1:
        for chunk in chunks:
            yield _validate_result(processor(chunk))
        return

    if max_pending is None:
        max_pending = workers * 2
    if not isinstance(max_pending, int) or isinstance(max_pending, bool) or max_pending < workers:
        raise ValueError("max_pending must be an integer greater than or equal to the worker count")

    executor_type = ThreadPoolExecutor if backend == "thread" else ProcessPoolExecutor
    with executor_type(max_workers=workers) as executor:
        yield from _parallel_results(chunks, processor, executor, max_pending)


def process_chunks(
    source: DataSource,
    processor: ChunkProcessor,
    chunksize: int = 10_000,
    n_jobs: int = 1,
    backend: str = "thread",
    max_pending: Optional[int] = None,
    **read_kwargs: Any,
) -> pd.DataFrame:
    """Process all chunks and concatenate the results in input order.

    For an output that must remain streaming, consume
    :func:`iter_processed_chunks` directly and write each returned frame with
    a storage adapter instead of materialising the final DataFrame.
    """
    results = list(
        iter_processed_chunks(
            source,
            processor,
            chunksize=chunksize,
            n_jobs=n_jobs,
            backend=backend,
            max_pending=max_pending,
            **read_kwargs,
        )
    )
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)
