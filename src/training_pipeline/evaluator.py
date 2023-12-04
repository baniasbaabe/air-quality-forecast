from typing import Union

import pandas as pd
import polars as pl
from loguru import logger
from utilsforecast.evaluation import evaluate
from utilsforecast.processing import join


class StatsForecastEvaluator:
    """Evaluator class for StatsForecast models."""

    def __init__(self, metrics):
        self.metrics = metrics

    @staticmethod
    def merge_datasets(
        forecasts: Union[pd.DataFrame, pl.DataFrame],
        data_test: Union[pd.DataFrame, pl.DataFrame],
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Merges forecasts and test set for evaluation.

        Args:
            forecasts (Union[pd.DataFrame, pl.DataFrame]): Predictions dataframe.
            data_test (Union[pd.DataFrame, pl.DataFrame]): Test dataframe

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: Merged dataframe prepared
            for evaluation.
        """
        logger.info("Preparing dataset for evaluation...")
        return join(data_test, forecasts, on=["unique_id", "ds"], how="left")

    def evaluate(
        self,
        forecasts: Union[pd.DataFrame, pl.DataFrame],
        data_test: Union[pd.DataFrame, pl.DataFrame],
    ) -> Union[pd.DataFrame, pl.DataFrame]:
        """Evaluates the forecasts using the metrics specified in the
        constructor.

        Args:
            forecasts (Union[pd.DataFrame, pl.DataFrame]): Predictions dataframe.
            data_test (Union[pd.DataFrame, pl.DataFrame]): Test dataframe

        Returns:
            Union[pd.DataFrame, pl.DataFrame]: Evaluation dataframe.
        """
        logger.info("Calculating metrics for forecasts...")
        data_test = self.merge_datasets(forecasts, data_test)
        return evaluate(data_test, metrics=self.metrics)
