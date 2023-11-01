"""Running the training pipeline."""

from dotenv import load_dotenv
from evaluator import StatsForecastEvaluator
from experiment_logger import CometExperimentLogger
from models import StatsForecastModel
from splitter import TrainTestSplit
from statsforecast.models import AutoTheta
from statsforecast.utils import ConformalIntervals
from utils import utils
from utilsforecast.losses import mae


# TODO: When read option is arrowflight, it doesn't work because dt
# will get +00:00 at the end
def main():
    """Main function for running the training pipeline."""
    load_dotenv()

    hyper_params = {"h": 7}

    project = utils.hopsworks_login()
    fs = utils.hopsworks_get_feature_store(project)
    fg = utils.hopsworks_get_feature_group(fs)
    data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    train_test_splitter = TrainTestSplit()

    data_train, data_test = train_test_splitter.train_test_split(data)

    naive = AutoTheta(prediction_intervals=ConformalIntervals(h=7, n_windows=3))

    model = StatsForecastModel(naive, hyper_params)

    model.train(data_train)
    forecasts = model.predict().reset_index()

    evaluator = StatsForecastEvaluator([mae])
    evaluation = evaluator.evaluate(forecasts, data_test)

    comet_experiment_logger = CometExperimentLogger(model, hyper_params, evaluation)
    comet_experiment_logger.log_experiment()


if __name__ == "__main__":
    main()
