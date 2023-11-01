import time

import pandas as pd
from loguru import logger
from statsforecast import StatsForecast


class StatsForecastModel:
    """Class for training and predicting with StatsForecast models."""

    def __init__(self, model, levels=[80, 90], freq="H"):
        self.model = model
        self.levels = levels
        self.freq = freq

    def train(self, data_train: pd.DataFrame) -> StatsForecast:
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

        self.model_obj = self.model_obj.fit(data_train)
        logger.info(
            f"Training model '{self.model.alias}' finished \
            in {time.time() - start_time:.2f} seconds"
        )

        return self.model_obj

    def predict(self, h: int = 7) -> pd.DataFrame:
        """Predicts the next h steps.

        Args:
            h (int, optional): Forecast horizon. Defaults to 7.

        Returns:
            pd.DataFrame: Prediction dataframe (with confidence intervals)
        """
        return self.model_obj.predict(h=h, level=self.levels)
