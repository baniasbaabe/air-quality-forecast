import pandas as pd
from loguru import logger
from utilsforecast.evaluation import evaluate
from utilsforecast.processing import join


class StatsForecastEvaluator:
    """Evaluator class for StatsForecast models."""

    def __init__(self, metrics):
        self.metrics = metrics

    @staticmethod
    def merge_datasets(
        forecasts: pd.DataFrame, data_test: pd.DataFrame
    ) -> pd.DataFrame:
        """Merges forecasts and test set for evaluation.

        Args:
            forecasts (pd.DataFrame): Predictions dataframe.
            data_test (pd.DataFrame): Test dataframe

        Returns:
            pd.DataFrame: Merged dataframe prepared
            for evaluation.
        """
        logger.info("Preparing dataset for evaluation...")
        return join(data_test, forecasts, on=["unique_id", "ds"], how="left")

    def evaluate(
        self, forecasts: pd.DataFrame, data_test: pd.DataFrame
    ) -> pd.DataFrame:
        """Evaluates the forecasts using the metrics specified in the
        constructor.

        Args:
            forecasts (pd.DataFrame): Predictions dataframe.
            data_test (pd.DataFrame): Test dataframe

        Returns:
            pd.DataFrame: Evaluation dataframe.
        """
        logger.info("Calculating metrics for forecasts...")
        data_test = self.merge_datasets(forecasts, data_test)
        return evaluate(data_test, metrics=self.metrics)
