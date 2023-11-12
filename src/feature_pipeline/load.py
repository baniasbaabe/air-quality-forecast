import polars as pl
from hopsworks.project import Project
from hsfs.feature_group import FeatureGroup
from hsfs.feature_store import FeatureStore
from loguru import logger

from src import utils


class Loader:
    def __get_project(self) -> Project:
        """Login into Hopsworks programmatically with API key and project name.
        Just a call from the utils module.

        Returns:
            Project: Hopsworks Project
        """
        logger.info("Getting project...")
        return utils.hopsworks_login()

    def __get_feature_store(self, project: Project) -> FeatureStore:
        """Connect to Hopsworks Feature Store. Just a call from the utils
        module.

        Args:
            project (Project): Hopsworks Project

        Returns:
            FeatureStore: Hopsworks Feature Store from the project
        """
        logger.info("Getting feature store...")
        return utils.hopsworks_get_feature_store(project)

    def __clean_feature_group(self, fs: FeatureStore) -> None:
        fg = utils.hopsworks_get_feature_group(fs)
        fg.delete()

    def __get_feature_group(self, fs: FeatureStore) -> FeatureGroup:
        """Get or create a Feature Group (collection of conceptually related
        features). Just a call from the utils module.

        Args:
            fs (FeatureStore): Hopsworks Feature Store

        Returns:
            FeatureGroup: Hopsworks Feature Group for Air Quality Timeseries
        """
        logger.info("Clean existing feature group...")
        try:
            self.__clean_feature_group(fs)
        except:
            logger.info("No feature group to clean.")
        logger.info("Create or get feature group")
        return utils.hopsworks_get_feature_group(fs)

    def __insert_data(self, fg: FeatureGroup, df: pl.DataFrame) -> FeatureGroup:
        """Insert DataFrame into the Feature Group. If it's a Polars DataFrame,
        convert it to Pandas DataFrame first since Hopsworks doesn't support
        Polars yet.

        Args:
            fg (FeatureGroup): Hopsworks Feature Group
            df (pl.DataFrame): (Processed) DataFrame

        Returns:
            FeatureGroup: Hopsworks FeatureGroup
        """
        logger.info("Inserting data into feature store...")
        if isinstance(df, pl.DataFrame):
            logger.info(
                "Polars dataframe detected. Converting \
                to Pandas dataframe since Hopsworks \
                    doesn't support Polars (yet)..."
            )
            df = df.to_pandas()
        fg.insert(df, wait=True)
        return fg

    def load_features(self, df: pl.DataFrame) -> None:
        """Method for running all the steps for loading data into Hopsworks.

        Args:
            df (pl.DataFrame): Processed Polars DataFrame
        """
        logger.info("Loading features into Hopsworks started...")
        project = self.__get_project()
        fs = self.__get_feature_store(project)
        fg = self.__get_feature_group(fs)
        fg = self.__insert_data(fg, df)
        logger.info("Loading features into Hopsworks finished.")
