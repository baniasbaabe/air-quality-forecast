import datetime
from typing import Tuple, Union

import pandas as pd
import polars as pl
from loguru import logger
from utilsforecast.processing import sort


class TrainTestSplit:
    """Time Series Splitter for splitting data into train and test sets."""

    def __init__(
        self,
        cutoff_hours: float = 168.0,
        time_col: str = "ds",
    ) -> None:
        self.cuttoff_hours = cutoff_hours
        self.time_col = time_col

    @staticmethod
    def _polars_split(group) -> None:
        cutoff_ds = group.select(pl.col("ds").max()).item() - datetime.timedelta(
            hours=168
        )
        train_mask = group.filter(pl.col("ds") <= (cutoff_ds)).with_columns(
            pl.lit("train").alias("set")
        )
        test_mask = group.filter(pl.col("ds") > (cutoff_ds)).with_columns(
            pl.lit("test").alias("set")
        )
        return pl.concat([train_mask, test_mask], how="vertical_relaxed")

    def train_test_split(
        self, data: Union[pd.DataFrame, pl.DataFrame]
    ) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame]]:
        """Splits data into train and test sets based on a given cutoff time.

        Args:
            data (pd.DataFrame): Dataframe with time series data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and test set.
        """
        logger.info("Splitting data into train and test sets...")
        data = sort(data, ["unique_id", self.time_col])

        if isinstance(data, pd.DataFrame):

            def split_train_test(group):
                cutoff_time = pd.to_timedelta(168, unit="hours")
                test_mask = group["ds"] > group["ds"].max() - cutoff_time
                return group[~test_mask], group[test_mask]

            data_train, data_test = zip(
                *data.groupby("unique_id", group_keys=False, sort=False).apply(
                    split_train_test
                )
            )
            data_train = pd.concat(data_train, ignore_index=True)
            data_test = pd.concat(data_test, ignore_index=True)

        else:
            data = data.group_by("unique_id").map_groups(self._polars_split)
            data_train = data.filter(pl.col("set") == "train").drop("set")
            data_test = data.filter(pl.col("set") == "test").drop("set")

        training_start_datetime = data_train[self.time_col].min()
        training_end_datetime = data_train[self.time_col].max()
        testing_start_datetime = data_test[self.time_col].min()
        testing_end_datetime = data_test[self.time_col].max()

        if training_start_datetime == training_end_datetime:
            raise ValueError(
                "Training set is empty. Please check your cutoff_hours parameter."
            )

        logger.info(
            f"Training set from {training_start_datetime} to {training_end_datetime}"
        )
        logger.info(f"Test set from {testing_start_datetime} to {testing_end_datetime}")

        return data_train, data_test
