import os
from typing import Union

import pandas as pd
import polars as pl
from loguru import logger
from pymongo.mongo_client import MongoClient


class MongoDBDatabase:
    """Class for saving predictions to MongoDB Atlas in the cloud."""

    def __init__(self) -> None:
        self.uri = f"mongodb+srv://{os.environ.get('MONGO_DB_USER')}:{os.environ.get('MONGO_DB_PASSWORD')}@airquality.boyq2bn.mongodb.net/?retryWrites=true&w=majority"

    def connect(self) -> MongoClient:
        """Connects to MongoDB Atlas.

        Returns:
            MongoClient: MongoClient for interacting with MongoDB Atlas.
        """
        logger.info("Connecting to MongoDB Atlas...")
        return MongoClient(self.uri)

    def _drop_collection(self):
        """Drops the collection since the free tier of MongoDB Atlas allows to
        store only 512 MB."""
        logger.info("Dropping collection to clean storage...")
        self.client["AirQuality"].drop_collection("AirQualityForecasts")

    def _insert_many(self, predictions: Union[pd.DataFrame, pl.DataFrame]) -> None:
        """Inserts the prediction dataframe into MongoDB Atlas.

        Args:
            predictions (Union[pd.DataFrame, pl.DataFrame]): Prediction dataframe.
        """
        logger.info("Inserting predictions into MongoDB Atlas...")
        if isinstance(predictions, pd.DataFrame):
            predictions = predictions.to_dict("records")
        else:
            predictions = predictions.to_dicts()
        self.client["AirQuality"]["AirQualityForecasts"].insert_many(predictions)

    def close(self):
        """Closes the connection to the MongoDB Client."""
        logger.info("Closing connection to MongoDB Atlas...")
        self.client.close()

    def save_predictions(self, predictions: pd.DataFrame) -> None:
        """Saves the predictions to MongoDB Atlas.

        Args:
            predictions (pd.DataFrame): Prediction dataframe.
        """
        logger.info("Saving predictions to MongoDB Atlas...")
        self.client = self.connect()
        self._drop_collection()
        self._insert_many(predictions)
        self.close()
