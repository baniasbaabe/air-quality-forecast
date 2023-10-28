import pandas as pd
from utilsforecast.evaluation import evaluate
from loguru import logger


class StatsForecastEvaluator:
    def __init__(self, metrics):
        self.metrics = metrics

    @staticmethod
    def merge_datasets(forecasts, data_test):
        logger.info("Preparing dataset for evaluation...")
        return data_test.merge(forecasts, on=["unique_id", "ds"], how="left")

    def evaluate(self, forecasts, data_test):
        logger.info("Calculating metrics for forecasts...")
        data_test = self.merge_datasets(forecasts, data_test)
        return evaluate(data_test, metrics=self.metrics)
