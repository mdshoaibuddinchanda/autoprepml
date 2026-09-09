"""Tests for streaming orchestration."""

import pandas as pd

from autoprepml.storage import InMemoryStorageAdapter
from autoprepml.streaming import stream_process, write_stream


def test_stream_process_and_write_stream_use_adapters():
    frame = pd.DataFrame({"value": range(5)})
    output = list(
        stream_process(
            frame,
            lambda chunk: chunk.assign(value=chunk["value"] + 10),
            chunksize=2,
            n_jobs=2,
        )
    )
    assert pd.concat(output, ignore_index=True)["value"].tolist() == [10, 11, 12, 13, 14]

    storage = InMemoryStorageAdapter()
    write_stream(
        frame,
        "processed",
        lambda chunk: chunk.assign(value=chunk["value"] + 1),
        chunksize=2,
        output_adapter=storage,
    )
    assert storage.read("processed")["value"].tolist() == [1, 2, 3, 4, 5]


def test_stream_process_accepts_custom_dataframe_iterable():
    chunks = (pd.DataFrame({"value": [value]}) for value in range(3))
    processed = list(
        stream_process(
            chunks,
            lambda chunk: chunk.assign(value=chunk["value"] * 3),
            chunksize=10,
        )
    )

    assert [chunk["value"].iloc[0] for chunk in processed] == [0, 3, 6]
