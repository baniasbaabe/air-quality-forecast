import os
from typing import Union

import joblib
import pandas as pd
import polars as pl
from comet_ml import Experiment
from loguru import logger


class CometExperimentLogger:
    """Logging Experiments with Comet.ml."""

    def __init__(
        self,
        model,
        evaluation: Union[pd.DataFrame, pl.DataFrame],
        metrics: list,
        hyper_params: dict,
    ) -> None:
        self.model = model
        self.evaluation = evaluation
        self.metrics = metrics
        self.hyper_params = hyper_params
        self.experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name=os.getenv("COMET_PROJECT_NAME"),
            workspace=os.getenv("COMET_WORKSPACE"),
        )

    def _save_model_locally(self) -> None:
        """Saving the model locally so that it can be pushed to Comet.ml."""
        logger.info("Saving model as a .pkl file locally...")
        joblib.dump(self.model.model_obj, "model.pkl")

    def log_experiment(self) -> None:
        """Logging experiment to Comet.ml and pushing model to registry."""
        logger.info("Logging experiment to Comet.ml...")
        self.experiment.log_parameters(self.hyper_params)

        for metric in self.metrics:
            if isinstance(self.evaluation, pd.DataFrame):
                metric_result = self.evaluation.query("metric == @metric")[
                    self.model.model.alias
                ].mean()
            else:
                metric_result = (
                    self.evaluation.filter(pl.col("metric").eq(metric.__name__))
                    .select(pl.col(self.model.model.alias).mean())
                    .item()
                )
            logger.info(f"Metric: {metric.__name__}, Result: {metric_result}")
            self.experiment.log_metrics({metric: metric_result})

        if isinstance(self.evaluation, pd.DataFrame):
            self.experiment.log_table("evaluation_per_sid.json", self.evaluation)

        self._save_model_locally()

        self.experiment.log_model(self.model.model.alias, "./model.pkl")

        self.experiment.register_model(self.model.model.alias, status="Production")
