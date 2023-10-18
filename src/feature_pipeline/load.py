import polars as pl
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from loguru import logger

from src import utils


class Loader:
    def __get_project(self) -> Project:
        logger.info("Getting project...")
        return utils.hopsworks_login()

    def __get_feature_store(self, project: Project) -> FeatureStore:
        logger.info("Getting feature store...")
        return utils.hopsworks_get_feature_store(project)

    def __get_feature_group(self, fs: FeatureStore) -> FeatureGroup:
        logger.info("Getting feature group...")
        return utils.hopsworks_get_feature_group(fs)

    def __insert_data(self, fg, df: pl.DataFrame) -> FeatureGroup:
        logger.info("Inserting data into feature store...")
        if isinstance(df, pl.DataFrame):
            logger.info(
                "Polars dataframe detected. Converting to Pandas dataframe since Hopsworks doesn't support Polars yet..."
            )
            df = df.to_pandas()
        fg.insert(df)
        return fg

    def load_features(self, df: pl.DataFrame) -> None:
        logger.info("Loading features into Hopsworks started...")
        project = self.__get_project()
        fs = self.__get_feature_store(project)
        fg = self.__get_feature_group(fs)
        fg = self.__insert_data(fg, df)
