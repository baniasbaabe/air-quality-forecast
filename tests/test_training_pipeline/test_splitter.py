from datetime import datetime

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes

from src.training_pipeline.splitter import TrainTestSplit


@given(
    data_frames(
        columns=[
            column(name="y", elements=st.floats(allow_nan=True)),
            column(
                name="ds",
                elements=st.datetimes(
                    min_value=datetime(2021, 1, 1),
                    allow_imaginary=False,
                ),
            ),
        ],
        index=range_indexes(min_size=200, max_size=1000),
    ),
    st.floats(min_value=40.0, max_value=168.0),
)
@settings(max_examples=2)
def test_train_test_split(data, cutoff_hours):
    train_test_splitter = TrainTestSplit(cutoff_hours=cutoff_hours)

    data_train, data_test = train_test_splitter.train_test_split(data)

    assert isinstance(data_train, pd.DataFrame)
    assert isinstance(data_test, pd.DataFrame)

    assert data_train["ds"].min() == data["ds"].min()
    assert data_train["ds"].max() == data["ds"].max()

    assert data_test["ds"].min() == data["ds"].min()
    assert data_test["ds"].max() == data["ds"].max()
