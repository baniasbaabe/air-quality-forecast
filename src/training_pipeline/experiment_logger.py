import os

import joblib
import pandas as pd
from comet_ml import Experiment
from loguru import logger


class CometExperimentLogger:
    """Logging Experiments with Comet.ml."""

    def __init__(self, model, hyper_params: dict, evaluation: pd.DataFrame) -> None:
        self.model = model
        self.evaluation = evaluation
        self.hyper_params = hyper_params
        self.experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
        )

    def __save_model_locally(self) -> None:
        """Saving the model locally so that it can be pushed to Comet.ml."""
        logger.info("Saving model as a .pkl file locally...")
        joblib.dump(self.model.model_obj, "model.pkl")

    def log_experiment(self) -> None:
        """Logging experiment to Comet.ml and pushing model to registry."""
        logger.info("Logging experiment to Comet.ml...")
        self.experiment.log_parameters(self.hyper_params)

        self.experiment.log_metrics({"mae": self.evaluation["AutoTheta"].mean()})

        self.experiment.log_table("evaluation_per_sid.json", self.evaluation)

        self.__save_model_locally()

        self.experiment.log_model("AutoTheta", "./model.pkl")

        self.experiment.register_model("AutoTheta", status="Production")
