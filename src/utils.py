import os

import hopsworks
import pandas as pd
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore


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
