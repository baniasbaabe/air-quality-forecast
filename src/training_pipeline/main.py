import pandas as pd
from dotenv import load_dotenv
from statsforecast import StatsForecast
from utilsforecast.evaluation import evaluate
from utilsforecast.losses import mape, mase, mae
from statsforecast.models import AutoTheta, Naive
from statsforecast.utils import ConformalIntervals
from src.utils import *
import datetime
import comet_ml
from comet_ml import Experiment
import joblib
from splitter import TrainTestSplit
from models import StatsForecastModel
from statsforecast.models import AutoTheta, Naive
from evaluator import StatsForecastEvaluator
from experiment_logger import CometExperimentLogger


# TODO: When read option is arrowflight, it doesn't work because dt will get +00:00 at the end
def main():
    load_dotenv()

    hyper_params = {"h": 7}

    project = hopsworks_login()
    fs = hopsworks_get_feature_store(project)
    fg = hopsworks_get_feature_group(fs)
    data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    train_test_splitter = TrainTestSplit()

    data_train, data_test = train_test_splitter.train_test_split(data)

    naive = Naive(
            prediction_intervals=ConformalIntervals(
                h=7, n_windows=3
            )
        )

    model = StatsForecastModel(naive, hyper_params)

    model.train(data_train)
    forecasts = model.predict().reset_index()

    # Forecasts has the columns ['unique_id', 'ds', 'AutoTheta', 'AutoTheta-lo-80', 'AutoTheta-hi-80', 'AutoTheta-lo-90', 'AutoTheta-hi-90', 'Naive', 'Naive-lo-80', 'Naive-hi-80', 'Naive-lo-90', 'Naive-hi-90'] where ds is timestamp

    evaluator = StatsForecastEvaluator([mae])
    evaluation = evaluator.evaluate(forecasts, data_test)


    comet_experiment_logger = CometExperimentLogger(model, hyper_params, evaluation)
    comet_experiment_logger.log_experiment()


if __name__ == "__main__":
    main()
