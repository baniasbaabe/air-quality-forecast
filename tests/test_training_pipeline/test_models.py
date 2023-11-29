import pandas as pd
import pytest
from statsforecast.models import Naive

from src.training_pipeline.models import StatsForecastModel


@pytest.fixture
def data_train():
    return pd.DataFrame(
        {
            "unique_id": ["1", "1", "2", "2"],
            "ds": ["2020-01-01", "2020-01-02", "2020-01-01", "2020-01-02"],
            "y": [1, 2, 3, 4],
        }
    )


@pytest.fixture
def mock_statsforecast(mocker):
    mock = mocker.Mock()
    mock.alias = "test_model"
    return mock


@pytest.fixture
def statsforecast_model(mock_statsforecast):
    return StatsForecastModel(mock_statsforecast, levels=[80])


def test_train_called(statsforecast_model, mock_statsforecast, data_train):
    statsforecast_model.model_obj = mock_statsforecast

    statsforecast_model.train(data_train)

    assert statsforecast_model._fitted is True
    assert mock_statsforecast.fit.called_once_with(data_train)


def test_predict_called(statsforecast_model, mock_statsforecast):
    statsforecast_model.model_obj = mock_statsforecast

    _ = statsforecast_model.predict(h=7)

    mock_statsforecast.predict.assert_called_once_with(h=7, level=[80])


def test_predict(data_train):
    naive_model = Naive()
    statsforecast_model = StatsForecastModel(naive_model, levels=[80])

    statsforecast_model.train(data_train)

    predictions = statsforecast_model.predict(h=2)

    assert set(predictions.columns.tolist()) == set(
        ["unique_id", "ds", "Naive", "Naive-lo-80", "Naive-hi-80"]
    )
