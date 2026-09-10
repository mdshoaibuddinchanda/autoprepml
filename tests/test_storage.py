"""Tests for storage adapters."""

import pandas as pd
import pytest

from autoprepml.storage import (
    FsspecStorageAdapter,
    InMemoryStorageAdapter,
    LocalStorageAdapter,
    get_storage_adapter,
)


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


def test_fsspec_memory_filesystem_round_trip():
    pytest.importorskip("fsspec")
    frame = pd.DataFrame({"value": [1, 2], "label": ["a", "b"]})
    storage = FsspecStorageAdapter()
    destination = "memory://autoprepml-tests/result.csv"

    storage.write(frame, destination)

    pd.testing.assert_frame_equal(storage.read(destination), frame)
    chunks = list(storage.iter_chunks(destination, chunksize=1))
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), frame)
