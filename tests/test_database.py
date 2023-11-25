from unittest.mock import MagicMock

import pandas as pd
import pytest
from pymongo.mongo_client import MongoClient

from src.database import MongoDBDatabase


@pytest.fixture
def mock_mongo_client():
    return MagicMock(spec=MongoClient)


@pytest.fixture
def mongo_db_instance(mock_mongo_client, monkeypatch):
    monkeypatch.setattr("src.database.MongoClient", mock_mongo_client)
    return MongoDBDatabase()


def test_connect(mongo_db_instance, mock_mongo_client):
    mongo_db_instance.connect()
    mock_mongo_client.assert_called_once_with(mongo_db_instance.uri)


def test_close(mongo_db_instance, mock_mongo_client):
    mongo_db_instance.client = mock_mongo_client.return_value
    mongo_db_instance.close()
    mock_mongo_client.return_value.close.assert_called_once()


def test_drop_collection(mongo_db_instance, mock_mongo_client):
    mongo_db_instance.client = mock_mongo_client.return_value
    mongo_db_instance._drop_collection()
    mongo_db_instance.client["AirQuality"].drop_collection.assert_called_once_with(
        "AirQualityForecasts"
    )


def test_insert_many(mongo_db_instance, mock_mongo_client):
    dummy_data = {"column1": [1, 2, 3], "column2": ["a", "b", "c"]}
    dummy_df = pd.DataFrame(dummy_data)

    mongo_db_instance.client = mock_mongo_client.return_value
    mongo_db_instance._insert_many(dummy_df)

    mongo_db_instance.client["AirQuality"][
        "AirQualityForecasts"
    ].insert_many.assert_called_once_with(dummy_df.to_dict("records"))
