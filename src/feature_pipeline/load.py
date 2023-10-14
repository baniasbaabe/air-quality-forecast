from loguru import logger
import hopsworks
import polars as pl
import os
from hopsworks.project import Project

from hsfs.feature_store import FeatureStore

from hsfs.feature_group import FeatureGroup


class Loader:
    def __get_project(self) -> Project:
        logger.info("Getting project...")
        return hopsworks.login(
            api_key_value=os.getenv("HOPSWORKS_API_KEY"),
            project=os.getenv("HOPSWORKS_PROJECT_NAME"),
        )

    def __get_feature_store(self, project: Project) -> FeatureStore:
        logger.info("Getting feature store...")
        return project.get_feature_store()

    def __get_feature_group(self, fs: FeatureStore) -> FeatureGroup:
        logger.info("Getting feature group...")
        return fs.get_or_create_feature_group(
            name="air_quality_timeseries",
            version=1,
            description="Timeseries for every sensor ID",
            event_time="dt",
            primary_key=["SID"],
        )

    def __insert_data(self, fg, df: pl.DataFrame) -> FeatureGroup:
        logger.info("Inserting data into feature store...")
        if isinstance(df, pl.DataFrame):
            logger.info("Polars dataframe detected. Converting to Pandas dataframe...")
            df = df.to_pandas()
        fg.insert(df)
        return fg

    def load_features(self, df: pl.DataFrame) -> None:
        logger.info("Loading features started...")
        project = self.__get_project()
        fs = self.__get_feature_store(project)
        fg = self.__get_feature_group(fs)
        fg = self.__insert_data(fg, df)
