import datetime
from typing import Tuple

import pandas as pd
from loguru import logger


class TrainTestSplit:
    """Time Series Splitter for splitting data into train and test sets."""

    def __init__(
        self,
        cutoff_hours: float = 168.0,
        time_col: str = "ds",
    ) -> None:
        self.cuttoff_hours = cutoff_hours
        self.time_col = time_col

    def train_test_split(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Splits data into train and test sets based on a given cutoff time.

        Args:
            data (pd.DataFrame): Dataframe with time series data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test set.
        """
        logger.info("Splitting data into train and test sets...")
        data_train = data[
            data["ds"]
            <= data["ds"].max() - datetime.timedelta(hours=self.cuttoff_hours)
        ]
        data_test = data[
            data["ds"] > data["ds"].max() - datetime.timedelta(hours=self.cuttoff_hours)
        ]

        training_start_datetime = data_train[self.time_col].min()
        training_end_datetime = data_train[self.time_col].max()
        testing_start_datetime = data_test[self.time_col].min()
        testing_end_datetime = data_test[self.time_col].max()

        logger.info(
            f"Train set from {training_start_datetime} to {training_end_datetime}"
        )
        logger.info(f"Test set from {testing_start_datetime} to {testing_end_datetime}")
        return data_train, data_test
