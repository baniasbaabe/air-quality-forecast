import math
from datetime import datetime

import hypothesis.strategies as st
import pytest
from hypothesis import given, settings
from hypothesis.extra.pandas import column, data_frames, range_indexes
from statsforecast.models import HoltWinters
from utilsforecast.losses import mae, mse

from src.utils import (
    calculate_required_size_for_conformal_prediction,
    load_statsforecast_model_class,
    load_utilsforecast_evaluation_function,
    postprocess_predictions,
)

positive_integers = st.integers(min_value=1)


def test_load_utilsforecast_evaluation_function():
    result = load_utilsforecast_evaluation_function(["mse", "mae"])

    assert result == [mse, mae]


def test_load_utilsforecast_evaluation_function_unknown_metric():
    with pytest.raises(AttributeError):
        _ = load_utilsforecast_evaluation_function(["unknown"])


def test_load_statsforecast_model_class():
    result = load_statsforecast_model_class("HoltWinters")

    assert result == HoltWinters


def test_load_statsforecast_model_class_unknown_model():
    with pytest.raises(AttributeError):
        _ = load_statsforecast_model_class("unknown")


@given(
    data_frames(
        columns=[
            column(name="unique_id", elements=st.text()),
            column(
                name="ds",
                elements=st.datetimes(
                    min_value=datetime(2021, 1, 1),
                    max_value=datetime(2022, 1, 30),
                    allow_imaginary=False,
                ),
            ),
            column(name="Model", elements=st.floats()),
            column(name="Model-lo-90", elements=st.floats()),
            column(name="Model-hi-90", elements=st.floats()),
        ],
        index=range_indexes(min_size=1, max_size=100),
    ),
)
@settings(max_examples=10)
def test_postprocess_predictions(df):
    df = postprocess_predictions(df, "Model")

    assert (df.select_dtypes(include="number") >= 0.0).all()


@given(
    h=positive_integers,
    n_windows=positive_integers,
    cutoff_hours=positive_integers,
)
def test_calculate_required_size_for_conformal_prediction(h, n_windows, cutoff_hours):
    expected_result = h * n_windows + cutoff_hours

    result = calculate_required_size_for_conformal_prediction(
        h, n_windows, cutoff_hours
    )

    assert result == expected_result
