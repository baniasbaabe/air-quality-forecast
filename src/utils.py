import importlib
import os
from typing import Any, List, Type

import hopsworks
import pandas as pd
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from loguru import logger


def hopsworks_login() -> Project:
    """Login into Hopsworks programmatically with API key and project name.

    Returns:
        Project: Hopsworks Project
    """
    return hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )


def hopsworks_get_feature_store(project: Project) -> FeatureStore:
    """Connect to Hopsworks Feature Store.

    Args:
        project (Project): Hopsworks Project

    Returns:
        FeatureStore: Hopsworks Feature Store from the project
    """
    return project.get_feature_store()


def hopsworks_get_feature_group(fs: FeatureStore) -> FeatureGroup:
    """Get or create a Feature Group (collection of conceptually related
    features).

    Args:
        fs (FeatureStore): Hopsworks Feature Store

    Returns:
        FeatureGroup: Hopsworks Feature Group for Air Quality Timeseries
    """
    return fs.get_or_create_feature_group(
        name="air_quality_timeseries",
        version=1,
        description="Timeseries for every sensor ID",
        event_time="dt",
        primary_key=["SID"],
    )


def hopsworks_get_features(fg: FeatureGroup) -> pd.DataFrame:
    """Select all features from a Feature Group without filters.

    Args:
        fg (FeatureGroup): Hopsworks Feature Group

    Returns:
        pd.DataFrame: DataFrame with all features from the Feature Group
    """
    return fg.select_all().read()


def load_statsforecast_model_class(
    model_name: str, base_module_path: str = "statsforecast.model"
) -> Type[Any]:
    try:
        module = importlib.import_module("statsforecast.models")
    except ValueError:
        logger.error(f"Unknown model: {model_name}")
        raise ValueError

    # Get the class from the module
    classifier_class = getattr(module, model_name)

    return classifier_class


def load_utilsforecast_evaluation_function(
    evaluation_names: List[str], base_module_path: str = "utilsforecast.losses"
) -> List[Any]:
    evaluation_classes = []
    for evaluation_name in evaluation_names:
        try:
            module = importlib.import_module(base_module_path)
        except ValueError:
            logger.warning(f"Unknown evaluation metric: {evaluation_name}")

        # Get the class from the module
        classifier_class = getattr(module, evaluation_name)
        evaluation_classes.append(classifier_class)

    return evaluation_classes


if __name__ == "__main__":
    # print(load_utilsforecast_evaluation_function(["mae", "mse"]))
    import yaml

    config = yaml.safe_load(open(r"config\config.yaml"))
    print(config)
