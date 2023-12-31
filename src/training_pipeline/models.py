import time
from typing import Union

import pandas as pd
import polars as pl
from loguru import logger
from statsforecast import StatsForecast


class StatsForecastModel:
    """Class for training and predicting with StatsForecast models."""

    def __init__(self, model, levels=[80, 90], freq="H"):
        self.model = model
        self.levels = levels
        self.freq = freq
        self._fitted = False

    def train(self, data_train: Union[pd.DataFrame, pl.DataFrame]) -> StatsForecast:
        """Train Statsforecast model.

        Args:
            data_train (_type_): _description_

        Returns:
            StatsForecast: StatsForecast class for running models.
        """
        logger.info(f"Training model '{self.model.alias}' started...")
        start_time = time.time()

        self.model_obj = StatsForecast(
            df=data_train,
            models=[self.model],
            freq=self.freq,
        )
        self._fitted = True

        self.model_obj = self.model_obj.fit(data_train)
        logger.info(
            f"Training model '{self.model.alias}' finished \
            in {time.time() - start_time:.2f} seconds"
        )

        return self.model_obj

    def predict(self, h: int = 7) -> Union[pd.DataFrame, pl.DataFrame]:
        """Predicts the next h steps.

        Args:
            h (int, optional): Forecast horizon. Defaults to 7.

        Returns:
           Union[pd.DataFrame, pl.DataFrame]: Prediction
           dataframe (with confidence intervals)
        """
        logger.info("Prediction started...")
        prediction = self.model_obj.predict(h=h, level=self.levels)
        if isinstance(prediction, pd.DataFrame):
            return prediction.reset_index()
        else:
            return prediction
