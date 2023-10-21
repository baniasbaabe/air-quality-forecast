import pandas as pd
from typing import Optional
from loguru import logger


class TrainTestSplit:
    def __init__(
        self, cutoff_hours: Optional[int] = 168, time_col: Optional[str] = "ds"
    ) -> tuple(pd.DataFrame, pd.DataFrame):
        cuttoff_hours = cutoff_hours

    def train_test_split(self, data: pd.DataFrame) -> pd.DataFrame:
        logger.info("Splitting data into train and test sets...")
        data_train = data[
            data["ds"] <= pd.Timestamp.now() - datetime.timedelta(hours=cuttoff_hours)
        ]
        data_test = data[
            data["ds"] > pd.Timestamp.now() - datetime.timedelta(hours=cutoff_hours)
        ]

        training_start_datetime = data_train[time_col].min()
        training_end_datetime = data_train[time_col].max()
        testing_start_datetime = data_test[time_col].min()
        testing_end_datetime = data_test[time_col].max()

        logger.info(
            f"Train set from {training_start_datetime} to {training_end_datetime}"
        )
        logger.info(f"Test set from {testing_start_datetime} to {testing_end_datetime}")
        return data_train, data_test
