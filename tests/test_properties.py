"""Property tests for invariants that should hold across varied inputs."""

import pandas as pd
from hypothesis import given, settings, strategies as st

from autoprepml import DataPlan
from autoprepml.batch import iter_chunks


@given(
    values=st.lists(st.integers(min_value=-1000, max_value=1000), min_size=0, max_size=100),
    chunksize=st.integers(min_value=1, max_value=25),
)
@settings(max_examples=40, deadline=None)
def test_chunking_preserves_order_and_row_count(values, chunksize):
    frame = pd.DataFrame({"value": values})

    chunks = list(iter_chunks(frame, chunksize=chunksize))

    if values:
        rebuilt = pd.concat(chunks, ignore_index=True)
    else:
        rebuilt = pd.DataFrame({"value": pd.Series(dtype=frame["value"].dtype)})
    pd.testing.assert_frame_equal(rebuilt, frame)
    assert sum(len(chunk) for chunk in chunks) == len(frame)
    assert all(len(chunk) <= chunksize for chunk in chunks)


@given(
    values=st.lists(
        st.tuples(st.integers(min_value=0, max_value=1000), st.sampled_from([0, 1])),
        min_size=4,
        max_size=30,
    )
)
@settings(max_examples=25, deadline=None)
def test_data_plan_transform_does_not_mutate_state(values):
    frame = pd.DataFrame(values, columns=["value", "label"])
    plan = DataPlan.infer(frame, target="label", fingerprint_mode="sampled").fit(frame)
    before = plan.state_fingerprint

    plan.transform(frame.drop(columns="label"))

    assert plan.state_fingerprint == before
