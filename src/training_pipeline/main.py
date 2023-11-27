"""Running the training pipeline."""

from pathlib import Path

import yaml
from dotenv import load_dotenv
from evaluator import StatsForecastEvaluator
from experiment_logger import CometExperimentLogger
from models import StatsForecastModel
from splitter import TrainTestSplit
from statsforecast.utils import ConformalIntervals

from src import utils


def main():
    """Main function for running the training pipeline."""
    load_dotenv()
    CONFIG = yaml.safe_load(open(Path("config/config.yaml")))

    project = utils.hopsworks_login()
    fs = utils.hopsworks_get_feature_store(project)
    fg = utils.hopsworks_get_feature_group(fs)
    data = fg.select(["sid", "dt", "p1"]).read(read_options={"use_hive": True})

    data = data.rename(columns={"sid": "unique_id", "dt": "ds", "p1": "y"})

    train_test_splitter = TrainTestSplit()

    data_train, data_test = train_test_splitter.train_test_split(data)

    model_class = utils.load_statsforecast_model_class(
        CONFIG["model"], CONFIG["statsforecast_module_path"]
    )

    sf_model = model_class(
        season_length=CONFIG["hyper_params"]["season_length"],
        prediction_intervals=ConformalIntervals(
            h=CONFIG["hyper_params"]["h"],
            n_windows=CONFIG["conformal_prediction"]["n_windows"],
        ),
    )

    model = StatsForecastModel(
        sf_model,
    )
    model.train(data_train.sort_values(by="ds"))
    forecasts = model.predict(h=CONFIG["hyper_params"]["h"])
    del model

    # Retrain
    model = StatsForecastModel(
        sf_model, levels=CONFIG["conformal_prediction"]["levels"], freq=CONFIG["freq"]
    )
    model.train(data)

    evaluation_metrics = utils.load_utilsforecast_evaluation_function(
        CONFIG["evaluation"]["metrics"], CONFIG["utilsforecast_module_path"]
    )

    evaluator = StatsForecastEvaluator(evaluation_metrics)
    evaluation = evaluator.evaluate(forecasts, data_test)

    comet_experiment_logger = CometExperimentLogger(
        model,
        evaluation,
        evaluation_metrics,
        CONFIG["hyper_params"],
    )
    comet_experiment_logger.log_experiment()


if __name__ == "__main__":
    main()
