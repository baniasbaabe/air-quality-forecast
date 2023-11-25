
import pandas as pd

from src.training_pipeline.experiment_logger import (
    CometExperimentLogger,
)


def test_comet_experiment_logger(mocker):
    experiment_instance_mock = mocker.patch(
        "src.training_pipeline.experiment_logger.Experiment"
    ).return_value
    mocked_joblib_dump = mocker.patch(
        "src.training_pipeline.experiment_logger.joblib.dump"
    )
    model_mock = mocker.Mock()
    model_mock.model.alias = "Naive"
    evaluation_data = pd.DataFrame(
        {"metric": ["mae", "mse"], model_mock.model.alias: [1.0, 2.0]}
    )
    metrics = ["mae", "mse"]
    hyper_params = {"param1": "value1", "param2": "value2"}

    model_instance_mock = mocker.Mock()
    model_instance_mock.model.alias = "Naive"
    experiment_instance_mock.log_metrics = mocker.Mock()

    model_mock.model_obj = model_instance_mock

    logger = CometExperimentLogger(model_mock, evaluation_data, metrics, hyper_params)

    logger.log_experiment()

    experiment_instance_mock.log_parameters.assert_called_once_with(hyper_params)

    for metric in metrics:
        metric_result = evaluation_data.query("metric == @metric")[
            model_mock.model.alias
        ].mean()
        experiment_instance_mock.log_metrics.assert_any_call({metric: metric_result})

    experiment_instance_mock.log_table.assert_called_once_with(
        "evaluation_per_sid.json", evaluation_data
    )

    mocked_joblib_dump.assert_called_once_with(model_instance_mock, "model.pkl")

    experiment_instance_mock.log_model.assert_called_once_with(
        model_mock.model.alias, "./model.pkl"
    )
    experiment_instance_mock.register_model.assert_called_once_with(
        model_mock.model.alias, status="Production"
    )
