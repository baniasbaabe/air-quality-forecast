from statsforecast.models import AutoTheta, Naive
from statsforecast.utils import ConformalIntervals
from statsforecast import StatsForecast
import time
from loguru import logger


class StatsForecastModel:
    def __init__(self, model, hyper_params, levels=[80, 90], freq="H"):
        self.model = model
        self.hyper_params = hyper_params
        self.levels = levels
        self.freq = freq

    def train(self, data_train):
        logger.info(f"Training model '{self.model.alias}' started...")
        start_time = time.time()

        self.model_obj = StatsForecast(
            df=data_train,
            models=[self.model],
            freq=self.freq,
        )

        self.model_obj = self.model_obj.fit(data_train)
        logger.info(f"Training model '{self.model.alias}' finished in {time.time() - start_time:.2f} seconds")

        return self.model_obj

    def predict(self, h=7):
        return self.model_obj.predict(h=h, level=self.levels)
