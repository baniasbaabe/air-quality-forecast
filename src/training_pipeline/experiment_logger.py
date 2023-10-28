import os
import comet_ml
from comet_ml import Experiment
from loguru import logger
import joblib


class CometExperimentLogger:
    def __init__(self, model, hyper_params, evaluation):
        self.model = model
        self.evaluation = evaluation
        self.hyper_params = hyper_params
        self.experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
        )

    def __save_model_locally(self):
        logger.info("Saving model as a .pkl file locally...")
        joblib.dump(self.model.model_obj, "model.pkl")

    def log_experiment(self):
        logger.info("Logging experiment to Comet.ml...")
        self.experiment.log_parameters(self.hyper_params)

        self.experiment.log_metrics({"mae": self.evaluation.loc[0]["Naive"]})

        self.experiment.log_table("evaluation_per_sid.json", self.evaluation)

        self.__save_model_locally()

        self.experiment.log_model("Naive", "./model.pkl")

        self.experiment.register_model("Naive", status="Production")
