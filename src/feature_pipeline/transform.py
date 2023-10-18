import polars as pl
from loguru import logger


class Transformation:
    def load_data_from_dict(self, data: dict) -> pl.DataFrame:
        logger.info("Loading data from json...")
        return pl.DataFrame(list(data["values"])).with_columns(SID=pl.lit(data["sid"]))

    def preprocess_dataframe(self, df: pl.DataFrame) -> pl.DataFrame:
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

    def resample_dataframe(self, df: pl.DataFrame, interval="1h") -> pl.DataFrame:
        logger.info("Resampling dataframe...")
        return df.groupby_dynamic(
            "dt", every="1h", by="SID", include_boundaries=False
        ).agg(pl.col(pl.Float32).mean())
