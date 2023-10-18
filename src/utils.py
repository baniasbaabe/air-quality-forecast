import os

import hopsworks
import pandas as pd
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore


def hopsworks_login() -> Project:
    return hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        project=os.getenv("HOPSWORKS_PROJECT_NAME"),
    )


def hopsworks_get_feature_store(project: Project) -> FeatureStore:
    return project.get_feature_store()


def hopsworks_get_feature_group(fs: FeatureStore) -> FeatureGroup:
    return fs.get_or_create_feature_group(
        name="air_quality_timeseries",
        version=1,
        description="Timeseries for every sensor ID",
        event_time="dt",
        primary_key=["SID"],
    )


def hopsworks_get_features(fg: FeatureGroup) -> pd.DataFrame:
    return fg.select_all().read()
