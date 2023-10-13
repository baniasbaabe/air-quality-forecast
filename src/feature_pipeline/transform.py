from abc import ABC, abstractmethod
import polars as pl
import loguru


class Transformation(ABC):
    @staticmethod
    def load_data_from_dict(data: dict) -> pl.DataFrame:
        loguru.info("Loading data from dictionary...")
        return pl.DataFrame(list(data["values"]))

    @staticmethod
    def preprocess_dataframe(df: pl.DataFrame) -> pl.DataFrame:
        return (
            df.with_columns(
                pl.col("dt").str.strptime(
                    pl.Datetime, "%Y-%m-%dT%H:%M:%S%.f%Z", strict=False
                )
            )
            .with_columns(SID=pl.lit(data["sid"]))
            .with_columns(pl.col("P1").str.parse_int())
            .with_columns(pl.col("P2").str.parse_int())
        )
