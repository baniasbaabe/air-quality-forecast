from loguru import logger
import hopsworks
import os


class Loader:
    def get_project(self) -> None:
        logger.info("Getting project...")
        return hopsworks.login(
            api_key_value=os.getenv["HOPSWORKS_API_KEY"],
            project=os.getenv["HOPSWORKS_PROJECT_NAME"],
        )

    def get_feature_store(self) -> None:
        logger.info("Getting feature store...")
        return project.get_feature_store()

    def get_feature_group(self) -> None:
        logger.info("Getting feature group...")
        return fs.get_or_create_feature_group(
            name="air_quality_timeseries",
            version=1,
            description="Timeseries for every sensor ID",
            event_time="dt",
        )

    def insert_data(self, fg, df: pl.DataFrame) -> None:
        logger.info("Inserting data into feature store...")
        fg.insert(df)
        return fg

    def update_description(self, fg):
        feature_descriptions = [
            {"name": "P1", "description": "P10 value"},
            {"name": "P2", "description": "P2.5 value"},
            {"name": "dt", "description": "Datetime of measurement"},
        ]

        for desc in feature_descriptions:
            fg.update_feature_description(desc["name"], desc["description"])

        return fg

    def load_features(self, df: pl.DataFrame):
        project = self.get_project()
        fs = self.get_feature_store()
        fg = self.get_feature_group()
        self.insert_data(fg, df)
        self.update_description(fg)
