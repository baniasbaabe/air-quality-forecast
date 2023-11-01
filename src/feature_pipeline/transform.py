import polars as pl
from loguru import logger


class Transformation:
    def __load_data_from_dict(self, data: dict) -> pl.DataFrame:
        """Creates a Polars dataframe from the data dictionary (fetched from
        API).

        Args:
            data (dict): Dictionary from the API

        Returns:
            pl.DataFrame: Polars DataFrame
        """
        logger.info("Loading data from json...")
        return pl.DataFrame(list(data["values"])).with_columns(SID=pl.lit(data["sid"]))

    def __preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
        """Preprocesses the DataFrame by casting the columns and format the
        datetime.

        Args:
            df (pl.DataFrame): Polars DataFrame

        Returns:
            pl.DataFrame: Processed DataFrame
        """
        logger.info("Preprocessing dataframe...")
        return df.with_columns(
            [
                pl.col("P1").cast(pl.Float32()),
                pl.col("P2").cast(pl.Float32()),
                pl.col("dt").str.strptime(
                    pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f%Z", strict=False
                ),
            ]
        )

    def __resample_dataframe(self, df: pl.DataFrame, interval="1h") -> pl.DataFrame:
        """Since sensor data is collected every ~3 minutes, we need to resample
        the data to a specific interval.

        Args:
            df (pl.DataFrame): DataFrame
            interval (str, optional): Chosen Interval. Defaults to "1h".

        Returns:
            pl.DataFrame: _description_
        """
        logger.info("Resampling dataframe...")
        return df.groupby_dynamic(
            "dt", every=interval, by="SID", include_boundaries=False
        ).agg(pl.col(pl.Float32).mean())

    def run(self, data: dict) -> pl.DataFrame:
        """Method for running all the steps.

        Args:
            data (dict): Sensor Data

        Returns:
            pl.DataFrame: Processed Polars DataFrame
        """
        logger.info("Running transformation...")
        df = self.__load_data_from_dict(data)
        df = self.__preprocess_dataframe(df)
        df = self.__resample_dataframe(df)
        return df
