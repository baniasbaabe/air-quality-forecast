import datetime
from functools import partial
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
        self.cutoff_hours = cutoff_hours
        self.time_col = time_col

    @staticmethod
    def _polars_split(group: pl.DataFrame, cutoff_hours: float = 168.0) -> pl.DataFrame:
        """User-defined function for splitting data into train and test sets,
        based on a given cutoff time for every unique_id.

        Args:
            group (pl.DataFrame): Group of data for a unique_id.
            cutoff_hours (float, optional): Cutoff value. Defaults to 168.0.

        Returns:
            pl.DataFrame: Dataframe labelled with train and test.
        """
        cutoff_ds = group.select(pl.col("ds").max()).item() - datetime.timedelta(
            hours=cutoff_hours
        )
        train_mask = group.filter(pl.col("ds") <= (cutoff_ds)).with_columns(
            pl.lit("train").alias("set")
        )
        test_mask = group.filter(pl.col("ds") > (cutoff_ds)).with_columns(
            pl.lit("test").alias("set")
        )
        return pl.concat([train_mask, test_mask], how="vertical_relaxed")

    @staticmethod
    def _pandas_split(
        group: pd.DataFrame, cutoff_hours: float = 168.0
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """User-defined function for splitting data into train and test sets,
        based on a given cutoff time for every unique_id.

        Args:
            group (pd.DataFrame): Group of data for a unique_id.
            cutoff_hours (float, optional): Cutoff value. Defaults to 168.0.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: Train and Test Dataframes.
        """
        cutoff_time = pd.to_timedelta(cutoff_hours, unit="hours")
        test_mask = group["ds"] > group["ds"].max() - cutoff_time
        return group[~test_mask], group[test_mask]

    def train_test_split(
        self, data: Union[pd.DataFrame, pl.DataFrame]
    ) -> Tuple[Union[pd.DataFrame, pl.DataFrame], Union[pd.DataFrame, pl.DataFrame]]:
        """Splits data into train and test sets based on a given cutoff time.

        Args:
            data (Union[pd.DataFrame, pl.DataFrame]): Dataframe with time series data.

        Returns:
            Tuple[Union[pd.DataFrame, pl.DataFrame],
            Union[pd.DataFrame, pl.DataFrame]]: Train and test set.
        """
        logger.info("Splitting data into train and test sets...")
        data = sort(data, ["unique_id", self.time_col])

        if isinstance(data, pd.DataFrame):
            split_function = partial(self._pandas_split, cutoff_hours=self.cutoff_hours)

            data_train, data_test = zip(
                *data.groupby("unique_id", group_keys=False, sort=False).apply(
                    split_function
                )
            )
            data_train = pd.concat(data_train, ignore_index=True)
            data_test = pd.concat(data_test, ignore_index=True)

        else:
            split_function = partial(self._polars_split, cutoff_hours=self.cutoff_hours)
            data = data.group_by("unique_id").map_groups(split_function)
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
