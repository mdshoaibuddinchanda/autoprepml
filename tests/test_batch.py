"""Tests for bounded chunk execution."""

import pandas as pd
import pytest

from autoprepml.batch import iter_chunks, iter_processed_chunks, process_chunks


def test_iter_chunks_dataframe_and_csv(tmp_path):
    frame = pd.DataFrame({"value": range(5)})
    assert [len(chunk) for chunk in iter_chunks(frame, chunksize=2)] == [2, 2, 1]

    source = tmp_path / "input.csv"
    frame.to_csv(source, index=False)
    chunks = list(iter_chunks(source, chunksize=2))
    pd.testing.assert_frame_equal(pd.concat(chunks, ignore_index=True), frame)


def test_process_chunks_preserves_order_with_threads():
    frame = pd.DataFrame({"value": range(8)})
    result = process_chunks(
        frame,
        lambda chunk: chunk.assign(double=chunk["value"] * 2),
        chunksize=2,
        n_jobs=2,
    )

    assert result["value"].tolist() == list(range(8))
    assert result["double"].tolist() == [value * 2 for value in range(8)]


def test_iter_processed_chunks_accepts_dataframe_iterators():
    chunks = (pd.DataFrame({"value": [value]}) for value in range(3))
    result = list(iter_processed_chunks(chunks, lambda chunk: chunk + 1, chunksize=10))

    assert [chunk["value"].iloc[0] for chunk in result] == [1, 2, 3]


def test_chunk_execution_validates_inputs():
    frame = pd.DataFrame({"value": [1]})
    with pytest.raises(ValueError, match="chunksize"):
        list(iter_chunks(frame, chunksize=0))
    with pytest.raises(ValueError, match="backend"):
        process_chunks(frame, lambda chunk: chunk, backend="gpu")
    with pytest.raises(TypeError, match="processor"):
        process_chunks(frame, None)
    with pytest.raises(TypeError, match="return"):
        process_chunks(frame, lambda chunk: chunk["value"].tolist())
