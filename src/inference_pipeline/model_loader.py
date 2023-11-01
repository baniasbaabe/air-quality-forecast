import os

import joblib
from comet_ml import API
from loguru import logger
from statsforecast import StatsForecast


class ModelLoader:
    """
    Class for downloading model from Comet.ml registry.
    """

    def __init__(self) -> None:
        self.api = API(os.environ.get("COMET_API_KEY"))

    def __get_current_model_version(self) -> str:
        """
        Get the current model version from the Comet.ml registry.
        The current model version is the latest model version
        that is in production.

        Returns:
            str: Current Model Version
        """
        logger.info("Getting current model version...")
        model_details = self.api.get_registry_model_details(
            os.environ.get("COMET_WORKSPACE"), "autotheta"
        )["versions"]
        model_versions = [
            md["version"] for md in model_details if md["status"] == "Production"
        ]
        return model_versions[-1]

    def __download_current_model(self) -> None:
        """
        Downloads the most current model from the Comet.ml registry.
        """
        logger.info("Downloading current model...")
        model_version = self.__get_current_model_version()
        self.api.download_registry_model(
            os.environ.get("COMET_WORKSPACE"),
            registry_name="autotheta",
            version=model_version,
            output_path="./",
            expand=True,
        )

    def load_production_model(self) -> StatsForecast:
        """
        Finds and downloads the most current model from the Comet.ml registry.

        Returns:
            StatsForecast: StatsForecast Class for running models.
        """
        logger.info("Loading production model...")
        self.__download_current_model()
        with open("./model.pkl", "rb") as f:
            model = joblib.load(f)
        return model
