"""Storage adapters for local, in-memory, and optional fsspec sources.

Adapters keep I/O concerns separate from preprocessing. The core package only
requires pandas; remote filesystems are supported through the optional
``fsspec`` dependency when a caller explicitly selects that adapter.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Iterable, Iterator, Optional, Union

import pandas as pd


DataSource = Union[str, Path, pd.DataFrame]


def _validate_chunksize(chunksize: int) -> None:
    if not isinstance(chunksize, int) or isinstance(chunksize, bool) or chunksize < 1:
        raise ValueError("chunksize must be a positive integer")


def _read_frame(path: Any, **read_kwargs: Any) -> pd.DataFrame:
    """Read a complete frame based on a path suffix."""
    name = str(path).lower()
    if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz")):
        return pd.read_csv(path, **read_kwargs)
    if name.endswith(".jsonl"):
        options = dict(read_kwargs)
        options.setdefault("lines", True)
        return pd.read_json(path, **options)
    if name.endswith(".json"):
        return pd.read_json(path, **read_kwargs)
    if name.endswith(".parquet"):
        return pd.read_parquet(path, **read_kwargs)
    raise ValueError("Unsupported data format; use CSV, JSON, JSONL, or Parquet")


class StorageAdapter(ABC):
    """Interface implemented by tabular storage backends."""

    @abstractmethod
    def read(self, source: Any, **read_kwargs: Any) -> pd.DataFrame:
        """Read a complete DataFrame."""

    @abstractmethod
    def iter_chunks(
        self, source: Any, chunksize: int = 10_000, **read_kwargs: Any
    ) -> Iterator[pd.DataFrame]:
        """Read a source as bounded DataFrame chunks."""

    @abstractmethod
    def write(self, frame: pd.DataFrame, destination: Any, **write_kwargs: Any) -> Any:
        """Write a DataFrame to a destination."""

    def write_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        destination: Any,
        **write_kwargs: Any,
    ) -> Any:
        """Write a stream of DataFrames, preserving row order."""
        frames = list(chunks)
        if not frames:
            return self.write(pd.DataFrame(), destination, **write_kwargs)
        if any(not isinstance(frame, pd.DataFrame) for frame in frames):
            raise TypeError("chunks must contain pandas DataFrame objects")
        return self.write(pd.concat(frames, ignore_index=True), destination, **write_kwargs)


class LocalStorageAdapter(StorageAdapter):
    """Read and atomically write local CSV, JSON, JSONL, and Parquet files."""

    def _path(self, source: Any) -> Path:
        if not isinstance(source, (str, Path)):
            raise TypeError("local storage sources must be paths")
        source_text = str(source).lower()
        if source_text.startswith(("s3://", "gs://", "az://", "https://")):
            raise ValueError("LocalStorageAdapter accepts local paths only")
        path = Path(source)
        return path

    def read(self, source: Any, **read_kwargs: Any) -> pd.DataFrame:
        """Read a complete local table."""
        path = self._path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input data source does not exist: {path}")
        return _read_frame(path, **read_kwargs)

    def iter_chunks(
        self, source: Any, chunksize: int = 10_000, **read_kwargs: Any
    ) -> Iterator[pd.DataFrame]:
        """Yield chunks using pandas streaming CSV reads where available."""
        _validate_chunksize(chunksize)
        path = self._path(source)
        if not path.exists():
            raise FileNotFoundError(f"Input data source does not exist: {path}")
        name = str(path).lower()
        if name.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz")):
            options = dict(read_kwargs)
            options.pop("chunksize", None)
            yield from pd.read_csv(path, chunksize=chunksize, **options)
            return
        frame = self.read(path, **read_kwargs)
        for start in range(0, len(frame), chunksize):
            yield frame.iloc[start : start + chunksize].copy()

    def write(self, frame: pd.DataFrame, destination: Any, **write_kwargs: Any) -> Path:
        """Atomically write a local table and return its path."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        path = self._path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = str(path).lower()
        with NamedTemporaryFile(
            mode="w+b", prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            if suffix.endswith((".csv", ".csv.gz", ".csv.bz2", ".csv.zip", ".csv.xz")):
                frame.to_csv(temporary_path, index=False, **write_kwargs)
            elif suffix.endswith((".json", ".jsonl")):
                options = dict(write_kwargs)
                if suffix.endswith(".jsonl"):
                    options.setdefault("orient", "records")
                    options.setdefault("lines", True)
                frame.to_json(temporary_path, **options)
            elif suffix.endswith(".parquet"):
                frame.to_parquet(temporary_path, index=False, **write_kwargs)
            else:
                raise ValueError("Unsupported data format; use CSV, JSON, JSONL, or Parquet")
            temporary_path.replace(path)
        finally:
            if temporary_path.exists():
                temporary_path.unlink()
        return path

    def write_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        destination: Any,
        **write_kwargs: Any,
    ) -> Path:
        """Stream CSV/JSONL output without materialising all chunks."""
        path = self._path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = str(path).lower()
        if not suffix.endswith((".csv", ".jsonl")):
            return super().write_chunks(chunks, path, **write_kwargs)

        temporary_path = None
        try:
            with NamedTemporaryFile(
                mode="w", prefix=f".{path.name}.", suffix=path.suffix, dir=path.parent, delete=False
            ) as temporary:
                temporary_path = Path(temporary.name)
                wrote = False
                for frame in chunks:
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("chunks must contain pandas DataFrame objects")
                    if suffix.endswith(".csv"):
                        frame.to_csv(temporary, index=False, header=not wrote, **write_kwargs)
                    else:
                        options = dict(write_kwargs)
                        options.setdefault("orient", "records")
                        options.setdefault("lines", True)
                        frame.to_json(temporary, **options)
                    wrote = True
            temporary_path.replace(path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
        return path


class InMemoryStorageAdapter(StorageAdapter):
    """Storage adapter useful for tests, notebooks, and service boundaries."""

    def __init__(self, initial: Optional[dict] = None):
        self._tables = {str(name): frame.copy() for name, frame in (initial or {}).items()}

    def read(self, source: Any, **read_kwargs: Any) -> pd.DataFrame:
        del read_kwargs
        if isinstance(source, pd.DataFrame):
            return source.copy()
        try:
            return self._tables[str(source)].copy()
        except KeyError as error:
            raise FileNotFoundError(f"In-memory table does not exist: {source}") from error

    def iter_chunks(
        self, source: Any, chunksize: int = 10_000, **read_kwargs: Any
    ) -> Iterator[pd.DataFrame]:
        _validate_chunksize(chunksize)
        frame = self.read(source, **read_kwargs)
        for start in range(0, len(frame), chunksize):
            yield frame.iloc[start : start + chunksize].copy()

    def write(self, frame: pd.DataFrame, destination: Any, **write_kwargs: Any) -> str:
        del write_kwargs
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        key = str(destination)
        self._tables[key] = frame.copy()
        return key


class FsspecStorageAdapter(LocalStorageAdapter):
    """Optional remote adapter for fsspec URLs such as S3 or GCS.

    Install ``fsspec`` and the filesystem-specific package, then pass this
    adapter explicitly. Remote writes use pandas/fsspec semantics and are not
    advertised as atomic across every object-store implementation.
    """

    def _filesystem(self, source: Any, mode: str = "rb"):
        try:
            import fsspec
        except ImportError as error:
            raise ImportError(
                "FsspecStorageAdapter requires the optional 'fsspec' package"
            ) from error
        return fsspec.open(str(source), mode=mode)

    def read(self, source: Any, **read_kwargs: Any) -> pd.DataFrame:
        if isinstance(source, pd.DataFrame):
            return source.copy()
        return _read_frame(source, **read_kwargs)

    def iter_chunks(
        self, source: Any, chunksize: int = 10_000, **read_kwargs: Any
    ) -> Iterator[pd.DataFrame]:
        _validate_chunksize(chunksize)
        name = str(source).lower()
        if name.endswith(".csv"):
            options = dict(read_kwargs)
            options.pop("chunksize", None)
            with self._filesystem(source, mode="rb") as handle:
                yield from pd.read_csv(handle, chunksize=chunksize, **options)
            return
        frame = self.read(source, **read_kwargs)
        for start in range(0, len(frame), chunksize):
            yield frame.iloc[start : start + chunksize].copy()

    def write(self, frame: pd.DataFrame, destination: Any, **write_kwargs: Any) -> str:
        """Write a remote table using fsspec's filesystem handle."""
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        name = str(destination).lower()
        if name.endswith((".csv", ".json", ".jsonl")):
            with self._filesystem(destination, mode="wt") as handle:
                if name.endswith(".csv"):
                    frame.to_csv(handle, index=False, **write_kwargs)
                else:
                    options = dict(write_kwargs)
                    if name.endswith(".jsonl"):
                        options.setdefault("orient", "records")
                        options.setdefault("lines", True)
                    frame.to_json(handle, **options)
        elif name.endswith(".parquet"):
            with self._filesystem(destination, mode="wb") as handle:
                frame.to_parquet(handle, index=False, **write_kwargs)
        else:
            raise ValueError("Unsupported data format; use CSV, JSON, JSONL, or Parquet")
        return str(destination)

    def write_chunks(
        self,
        chunks: Iterable[pd.DataFrame],
        destination: Any,
        **write_kwargs: Any,
    ) -> str:
        """Write remote CSV/JSONL streams without local temporary files."""
        name = str(destination).lower()
        if name.endswith((".csv", ".jsonl")):
            with self._filesystem(destination, mode="wt") as handle:
                wrote = False
                for frame in chunks:
                    if not isinstance(frame, pd.DataFrame):
                        raise TypeError("chunks must contain pandas DataFrame objects")
                    if name.endswith(".csv"):
                        frame.to_csv(handle, index=False, header=not wrote, **write_kwargs)
                    else:
                        options = dict(write_kwargs)
                        options.setdefault("orient", "records")
                        options.setdefault("lines", True)
                        frame.to_json(handle, **options)
                    wrote = True
            return str(destination)
        return super().write_chunks(chunks, destination, **write_kwargs)


def get_storage_adapter(
    source: Any = None, adapter: Optional[StorageAdapter] = None
) -> StorageAdapter:
    """Return an explicit adapter or a sensible local/in-memory default."""
    if adapter is not None:
        required_methods = ("read", "iter_chunks", "write", "write_chunks")
        if not all(callable(getattr(adapter, method, None)) for method in required_methods):
            raise TypeError("adapter must implement StorageAdapter")
        return adapter
    if isinstance(source, (str, Path)):
        return LocalStorageAdapter()
    return InMemoryStorageAdapter()
