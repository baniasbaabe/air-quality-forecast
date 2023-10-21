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
from models import StatsForecastTrainer


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

    # data_train = data[data["ds"] <= pd.Timestamp.now() - datetime.timedelta(days=7)]
    # data_test = data[data["ds"] > pd.Timestamp.now() - pd.to_timedelta("7day")]

    # data_train.to_csv("train.csv", index=False)
    # data_test.to_csv("test.csv", index=False)

    # theta = AutoTheta(prediction_intervals=ConformalIntervals(h=7, n_windows=3))
    # baseline = Naive(prediction_intervals=ConformalIntervals(h=7, n_windows=3))

    # sf = StatsForecast(
    #     df=data_train,
    #     models=[theta, baseline],
    #     freq="H",
    # )

    # levels = [80, 90]

    # forecasts = sf.forecast(**hyper_params, level=levels)
    # forecasts = forecasts.reset_index()

    statsforecast_trainer = StatsForecastTrainer(data_train, data_test, hyper_params)

    statsforecast_trainer.train()

    forecasts = statsforecast_trainer.predict(data_test).reset_index()

    # Forecasts has the columns ['unique_id', 'ds', 'AutoTheta', 'AutoTheta-lo-80', 'AutoTheta-hi-80', 'AutoTheta-lo-90', 'AutoTheta-hi-90', 'Naive', 'Naive-lo-80', 'Naive-hi-80', 'Naive-lo-90', 'Naive-hi-90'] where ds is timestamp

    # Evaluation:

    data_test = data_test.merge(forecasts, on=["unique_id", "ds"], how="left")

    # Evaluate Predictions for every model for every sensorid

    evaluation = evaluate(data_test, metrics=[mae])

    # Log Experiment to Comet.ml
    experiment = Experiment(
        api_key=os.getenv("COMET_API_KEY"),
        project_name=os.getenv("COMET_PROJECT_NAME"),
        workspace=os.getenv("COMET_WORKSPACE"),
    )

    joblib.dump(sf, "model.pkl")

    experiment.log_parameters(hyper_params)

    experiment.log_metrics({"mae": evaluation.loc[0]["AutoTheta"]})

    experiment.log_table("evaluation_per_sid.json", evaluation)

    experiment.log_model("AutoTheta", "./model.pkl")


if __name__ == "__main__":
    main()
