import polars as pl
from loguru import logger


class Transformation:
    @staticmethod
    def load_data_from_dict(data: dict) -> pl.DataFrame:
        logger.info("Loading data from dictionary...")
        return pl.DataFrame(list(data["values"])).with_columns(SID=pl.lit(data["sid"]))

    @staticmethod
    def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
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

    @staticmethod
    def resample_dataframe(df: pl.DataFrame, interval="1h") -> pl.DataFrame:
        logger.info("Resampling dataframe...")
        return df.groupby_dynamic(
            "dt", every="1h", by="SID", include_boundaries=False
        ).agg(pl.col(pl.Float32).mean())
