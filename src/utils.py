import importlib
import os
from typing import Any, List, Type, Union

import hopsworks
import pandas as pd
import polars as pl
import polars.selectors as cs
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
    model_name: str, base_module_path: str = "statsforecast.models"
) -> Type[Any]:
    """Dynamically load a statsforecast model class, given the model name.

    Args:
        model_name (str): Model Name (should be the same as the class name)
        base_module_path (str, optional): Module where the
        Model is located. Defaults to "statsforecast.model".

    Raises:
        AttributeError: When the model name is not found

    Returns:
        Type[Any]: Class of model
    """
    try:
        module = importlib.import_module(base_module_path)
    except AttributeError:
        logger.error(f"Unknown model: {model_name}")
        raise AttributeError

    # Get the class from the module
    classifier_class = getattr(module, model_name)

    return classifier_class


def load_utilsforecast_evaluation_function(
    evaluation_names: List[str], base_module_path: str = "utilsforecast.losses"
) -> List[Any]:
    """Dynamically loads utilsforecast metrics classes, given a list of metric
    names.

    Args:
        evaluation_names (List[str]): List of metric names (same as class names)
        base_module_path (str, optional): Module where metrics
        are located. Defaults to "utilsforecast.losses".

    Returns:
        List[Any]: List of metric class names
    """
    evaluation_classes = []
    for evaluation_name in evaluation_names:
        try:
            module = importlib.import_module(base_module_path)
        except AttributeError:
            logger.warning(f"Unknown evaluation metric: {evaluation_name}")
            raise AttributeError

        # Get the class from the module
        classifier_class = getattr(module, evaluation_name)
        evaluation_classes.append(classifier_class)

    return evaluation_classes


def calculate_required_size_for_conformal_prediction(h, n_windows, cutoff_hours):
    """Calculates the minimum required size for conformal prediction (Forecast
    Horizon + n_windows + (Cutoff Hours // WEEKDAYS))

    Returns:
        int: Minimum required size for conformal prediction
    """
    return h * n_windows + cutoff_hours


def filter_ids_for_conformal_prediction(
    df: pl.DataFrame, required_size: int = 22
) -> pl.DataFrame:
    """Filters all ids where count is bigger than the required size for
    conformal prediction (minimum required size: Forecast Horizon + n_windows +
    (Cutoff Hours // WEEKDAYS))

    Args:
        df (pl.DataFrame): Polars DataFrame

    Returns:
        pl.DataFrame: Filtered Polars DataFrame
    """
    return df.filter(
        pl.col("sid").is_in(
            df.groupby("sid")
            .agg(pl.count())
            .filter(pl.col("count") > required_size)
            .select(pl.col("sid"))
            .to_series()
            .to_list()
        )
    )


def postprocess_predictions(
    predictions: Union[pd.DataFrame, pl.DataFrame], model_name: str
) -> Union[pd.DataFrame, pl.DataFrame]:
    """Postprocess predictions by setting all negative values to 0 and changing
    Model name.

    Args:
        predictions (Union[pd.DataFrame, pl.DataFrame]): Prediction dataframe.

    Returns:
        Union[pd.DataFrame, pl.DataFrame]: Postprocessed prediction dataframe.
    """
    if isinstance(predictions, pd.DataFrame):
        predictions = predictions.reset_index()
        float_columns = predictions.select_dtypes(include="float").columns
        predictions[float_columns] = predictions[float_columns].applymap(
            lambda x: max(x, 0)
        )
    else:
        predictions.with_columns(
            cs.by_dtype(pl.NUMERIC_DTYPES).map_elements(lambda x: max(x, 0))
        )
    predictions.columns = [
        column.replace(model_name, "Model") for column in predictions.columns
    ]
    return predictions
