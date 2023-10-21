from statsforecast.models import AutoTheta, Naive
from statsforecast.utils import ConformalIntervals
from statsforecast import StatsForecast


class StatsForecastTrainer:
    def __init__(self, data_train, data_test, hyper_params, levels=[80, 90], freq="H"):
        self.data_train = data_train
        self.hyper_params = hyper_params
        self.levels = levels
        self.freq = freq
        self.models = models

    def train(self):
        theta = AutoTheta(
            prediction_intervals=ConformalIntervals(
                h=self.hyper_params["h"], n_windows=3
            )
        )
        baseline = Naive(
            prediction_intervals=ConformalIntervals(
                h=self.hyper_params["h"], n_windows=3
            )
        )

        sf = StatsForecast(
            df=self.data_train,
            models=[theta, baseline],
            freq=self.freq,
        )

        self.sf = sf.fit(**self.hyper_params)
        # forecasts = forecasts.reset_index()

        return sf

    def predict(self, data, h=7):
        return self.sf.predict(h=7, X_df=data, level=self.levels)
