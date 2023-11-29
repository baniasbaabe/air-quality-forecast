from datetime import datetime
from string import ascii_letters, digits

import polars as pl
from hypothesis import given, settings
from hypothesis import strategies as st
from polars.testing import parametric

from src.feature_pipeline.transform import Transformation

value_entry_strategy = st.builds(
    lambda P1, P2, dt: {"P1": P1, "P2": P2, "dt": dt},
    st.floats(min_value=0.0, max_value=100.0),
    st.floats(min_value=0.0, max_value=100.0),
    st.datetimes(
        min_value=datetime(2020, 1, 1),
        max_value=datetime(2025, 1, 1),
        allow_imaginary=False,
    ).map(lambda date: date.strftime("%Y-%m-%dT%H:%M:%S%.f%Z")),
)

sid_entry_strategy = st.builds(
    lambda sid, count, values: {"sid": sid, "count": count, "values": values},
    st.text(alphabet=ascii_letters + digits),
    st.integers(min_value=1, max_value=100),
    st.lists(value_entry_strategy, min_size=1),
)

data_structure_strategy = st.lists(sid_entry_strategy, min_size=1)


@given(data_structure_strategy)
@settings(max_examples=20)
def test_load_data_from_dict(data: dict):
    transformer = Transformation()

    loaded_data = transformer._load_data_from_dict(data)

    assert isinstance(loaded_data, pl.DataFrame)
    assert loaded_data.columns == ["sid", "count", "values"]
    assert loaded_data["values"].dtype == pl.List(pl.Struct)


@given(
    parametric.dataframes(
        cols=[
            parametric.column(
                "sid", strategy=st.text(min_size=1, alphabet=ascii_letters + digits)
            ),
            parametric.column("count", strategy=st.integers(min_value=1, max_value=10)),
            parametric.column(
                "values",
                strategy=st.lists(value_entry_strategy, min_size=1, max_size=1),
            ),
        ],
        min_size=1,
        max_size=5,
        lazy=False,
    )
)
@settings(max_examples=20)
def test_preprocess_dataframe(df):
    transformer = Transformation()

    df = transformer._preprocess_dataframe(df)

    assert isinstance(df, pl.DataFrame)
    assert df.columns == ["sid", "P1", "P2", "dt"]
    assert df["dt"].is_sorted() == True
