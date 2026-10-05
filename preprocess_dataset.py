import os
from pathlib import Path

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import polars as pl


def main():
    directory = Path("original_dataset")

    columns = [
        "lv_feeder_unique_id",
        "total_consumption_active_import",
        "aggregated_device_count_active",
        "data_collection_log_timestamp",
    ]
    df = pl.read_parquet(directory / "*.parquet", columns=columns)

    df = df.with_columns(
        (
            pl.col("total_consumption_active_import")
            / pl.col("aggregated_device_count_active")
        )
        .round(0)
        .cast(pl.Int64)
        .alias("mean_consumption_per_meter")
    )
    df = df.drop("total_consumption_active_import")
    df = df.drop("aggregated_device_count_active")

    df = df.with_columns(
        pl.col("data_collection_log_timestamp").cast(pl.Datetime("ms", "UTC"))
    )
    df = df.sort(["lv_feeder_unique_id", "data_collection_log_timestamp"])

    df.write_parquet("dataset.parquet")


if __name__ == "__main__":
    main()
