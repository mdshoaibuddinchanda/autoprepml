"""Tests for storage adapters."""

import io
from contextlib import contextmanager

import pandas as pd
import pytest

from autoprepml.storage import (
    FsspecStorageAdapter,
    InMemoryStorageAdapter,
    LocalStorageAdapter,
    get_storage_adapter,
    StorageAdapter,
    _format_for_path,
    _read_frame,
)
from autoprepml import storage as storage_module


def test_local_storage_reads_and_streams_csv(tmp_path):
    frame = pd.DataFrame({"value": range(5), "label": list("abcde")})
    source = tmp_path / "source.csv"
    destination = tmp_path / "destination.csv"
    frame.to_csv(source, index=False)
    storage = LocalStorageAdapter()

    pd.testing.assert_frame_equal(storage.read(source), frame)
    chunks = list(storage.iter_chunks(source, chunksize=2))
    storage.write_chunks(chunks, destination)
    pd.testing.assert_frame_equal(pd.read_csv(destination), frame)


def test_local_storage_writes_jsonl_and_compressed_csv(tmp_path):
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = LocalStorageAdapter()
    jsonl = tmp_path / "result.jsonl"
    compressed = tmp_path / "result.csv.gz"

    storage.write_chunks([frame.iloc[:1], frame.iloc[1:]], jsonl)
    storage.write(frame, compressed)

    pd.testing.assert_frame_equal(storage.read(jsonl), frame)
    pd.testing.assert_frame_equal(storage.read(compressed), frame)


@pytest.mark.parametrize("suffix", ["csv.bz2", "csv.zip", "csv.xz"])
def test_local_storage_round_trips_compressed_csv_variants(tmp_path, suffix):
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = LocalStorageAdapter()
    destination = tmp_path / f"result.{suffix}"

    storage.write(frame, destination)

    pd.testing.assert_frame_equal(storage.read(destination), frame)


def test_local_storage_supports_json_and_empty_streams(tmp_path):
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = LocalStorageAdapter()
    json_path = tmp_path / "result.json"
    empty_path = tmp_path / "empty.csv"

    storage.write(frame, json_path)
    storage.write_chunks([], empty_path)

    pd.testing.assert_frame_equal(storage.read(json_path), frame)
    assert storage.read(empty_path).empty
    assert len(list(storage.iter_chunks(empty_path))) == 1


def test_local_storage_rejects_invalid_chunks_and_extensions(tmp_path):
    storage = LocalStorageAdapter()
    with pytest.raises(ValueError, match="Unsupported data format"):
        storage.write(pd.DataFrame(), tmp_path / "result.txt")
    with pytest.raises(TypeError, match="DataFrame"):
        storage.write_chunks(["not-a-frame"], tmp_path / "result.csv")
    assert not (tmp_path / "result.csv").exists()


def test_memory_storage_round_trip_and_factory():
    frame = pd.DataFrame({"value": [1, 2, 3]})
    storage = InMemoryStorageAdapter()
    assert get_storage_adapter(frame).__class__ is InMemoryStorageAdapter
    assert get_storage_adapter("file.csv").__class__ is LocalStorageAdapter

    storage.write(frame, "table")
    pd.testing.assert_frame_equal(storage.read("table"), frame)
    assert [len(chunk) for chunk in storage.iter_chunks("table", chunksize=2)] == [2, 1]


def test_memory_storage_validates_initial_tables():
    with pytest.raises(TypeError, match="initial"):
        InMemoryStorageAdapter(initial=[])
    with pytest.raises(TypeError, match="DataFrame"):
        InMemoryStorageAdapter(initial={"bad": []})


def test_storage_rejects_bad_sources_and_adapters(tmp_path):
    storage = LocalStorageAdapter()
    with pytest.raises(FileNotFoundError):
        storage.read(tmp_path / "missing.csv")
    (tmp_path / "data.txt").write_text("not a table", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported data format"):
        list(storage.iter_chunks(tmp_path / "data.txt"))
    with pytest.raises(TypeError, match="adapter"):
        get_storage_adapter("file.csv", adapter=object())


def test_storage_format_and_parquet_dispatch(monkeypatch, tmp_path):
    frame = pd.DataFrame({"x": [1]})
    assert _format_for_path("table.parquet") == "parquet"
    with pytest.raises(ValueError, match="Unsupported"):
        _format_for_path("table.tsv")
    monkeypatch.setattr(pd, "read_parquet", lambda path, **kwargs: frame)
    pd.testing.assert_frame_equal(_read_frame(tmp_path / "table.parquet"), frame)


def test_storage_base_write_chunks_and_memory_write_validation(tmp_path):
    class MinimalStorage(StorageAdapter):
        def read(self, source, **kwargs):
            return pd.DataFrame()

        def iter_chunks(self, source, chunksize=1, **kwargs):
            yield pd.DataFrame()

        def write(self, frame, destination, **kwargs):
            self.last = frame
            return destination

    adapter = MinimalStorage()
    destination = tmp_path / "out.csv"
    assert adapter.write_chunks([], destination) == destination
    adapter.write_chunks([pd.DataFrame({"x": [1]}), pd.DataFrame({"x": [2]})], destination)
    assert adapter.last["x"].tolist() == [1, 2]
    memory = InMemoryStorageAdapter()
    with pytest.raises(TypeError, match="DataFrame"):
        memory.write([], "table")


def test_fsspec_memory_filesystem_round_trip():
    pytest.importorskip("fsspec")
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = FsspecStorageAdapter()
    destination = "memory://autoprepml-tests/result.csv"

    storage.write(frame, destination)

    pd.testing.assert_frame_equal(storage.read(destination), frame)
    chunks = list(storage.iter_chunks(destination, chunksize=1))
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), frame)


def test_storage_validates_chunk_sizes_and_memory_sources():
    frame = pd.DataFrame({"value": [1, 2]})
    memory = InMemoryStorageAdapter({"table": frame})
    with pytest.raises(ValueError, match="positive"):
        list(memory.iter_chunks("table", chunksize=0))
    with pytest.raises(ValueError, match="positive"):
        list(LocalStorageAdapter().iter_chunks("missing.csv", chunksize=True))
    pd.testing.assert_frame_equal(memory.read(frame), frame)
    with pytest.raises(FileNotFoundError, match="In-memory"):
        memory.read("missing")


def test_local_storage_remote_paths_are_rejected(tmp_path):
    storage = LocalStorageAdapter()
    for source in (
        "s3://bucket/file.csv",
        "gs://bucket/file.csv",
        "az://container/file.csv",
        "https://example/file.csv",
    ):
        with pytest.raises(ValueError, match="local paths"):
            storage._path(source)
    with pytest.raises(TypeError, match="paths"):
        storage._path(pd.DataFrame())


def test_local_storage_jsonl_stream_handles_empty_and_fallback_formats(tmp_path, monkeypatch):
    storage = LocalStorageAdapter()
    destination = tmp_path / "empty.jsonl"
    storage.write_chunks([], destination)
    assert destination.exists()
    assert storage.read(destination).empty

    captured = {}

    def fake_write(frame, path, **kwargs):
        captured["frame"] = frame.copy()
        captured["path"] = path
        captured["kwargs"] = kwargs
        return path

    monkeypatch.setattr(storage, "write", fake_write)
    result = storage.write_chunks([pd.DataFrame({"x": [1]})], tmp_path / "result.parquet")
    assert result.name == "result.parquet"
    assert captured["frame"]["x"].tolist() == [1]


def test_fsspec_adapter_paths_and_streaming_without_optional_dependency(monkeypatch):
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = FsspecStorageAdapter()

    @contextmanager
    def handle_for(path, mode="rb"):
        if "r" in mode:
            yield io.BytesIO(frame.to_csv(index=False).encode())
        else:
            yield io.StringIO()

    monkeypatch.setattr(storage, "_filesystem", handle_for)
    chunks = list(storage.iter_chunks("remote.csv", chunksize=1))
    assert [len(chunk) for chunk in chunks] == [1, 1]
    assert storage.write(frame, "remote.csv") == "remote.csv"
    assert storage.write(frame, "remote.jsonl") == "remote.jsonl"
    assert storage.write_chunks([frame.iloc[:1], frame.iloc[1:]], "remote.csv") == "remote.csv"
    assert storage.write_chunks([frame.iloc[:1]], "remote.jsonl") == "remote.jsonl"
    with pytest.raises(TypeError, match="DataFrame"):
        storage.write("not-a-frame", "remote.csv")
    with pytest.raises(TypeError, match="DataFrame"):
        storage.write_chunks(["not-a-frame"], "remote.csv")

    monkeypatch.setattr(storage_module, "_read_frame", lambda source, **kwargs: frame.copy())
    pd.testing.assert_frame_equal(storage.read("remote.json", unused=True), frame)
    pd.testing.assert_frame_equal(storage.read(frame), frame)
    non_csv = list(storage.iter_chunks("remote.json", chunksize=1))
    assert [len(chunk) for chunk in non_csv] == [1, 1]


def test_fsspec_adapter_reports_missing_optional_dependency(monkeypatch):
    storage = FsspecStorageAdapter()
    monkeypatch.setitem(__import__("sys").modules, "fsspec", None)
    with pytest.raises(ImportError, match="fsspec"):
        storage._filesystem("remote.csv")
