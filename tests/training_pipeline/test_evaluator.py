import pandas as pd
import pytest
from utilsforecast.losses import mae, mse

from src.training_pipeline.evaluator import StatsForecastEvaluator


@pytest.fixture
def forecasts_data():
    return pd.DataFrame(
        {
            "unique_id": [1, 2, 3],
            "ds": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "Naive": [10, 20, 30],
        }
    )


@pytest.fixture
def test_data():
    return pd.DataFrame(
        {
            "unique_id": [1, 2, 3],
            "ds": pd.to_datetime(["2023-01-01", "2023-01-02", "2023-01-03"]),
            "y": [11, 22, 33],
        }
    )


@pytest.fixture
def evaluator():
    metrics = [mae, mse]
    return StatsForecastEvaluator(metrics)


def test_merge_datasets(evaluator, forecasts_data, test_data):
    merged_data = evaluator.merge_datasets(forecasts_data, test_data)
    assert len(merged_data) == len(test_data)
    assert set(merged_data.columns) == set(["unique_id", "ds", "y", "Naive"])


def test_evaluate(evaluator, forecasts_data, test_data):
    evaluation_result = evaluator.evaluate(forecasts_data, test_data)
    assert isinstance(evaluation_result, pd.DataFrame)
    assert set(evaluation_result.columns) == set(["unique_id", "metric", "Naive"])
    assert set(evaluation_result["metric"].unique()) == set(["mae", "mse"])
