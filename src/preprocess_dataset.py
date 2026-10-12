import os
from datetime import timedelta
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
        "geometry",
    ]
    lf = pl.scan_parquet(directory / "*.parquet").select(columns)

    lf = lf.with_columns(
        (
            pl.col("total_consumption_active_import")
            / pl.col("aggregated_device_count_active")
        )
        .round(0)
        .cast(pl.Int64)
        .alias("mean_consumption_per_meter")
    )
    lf = lf.drop("total_consumption_active_import")

    lf = lf.with_columns(
        pl.col("data_collection_log_timestamp").cast(pl.Datetime("ms", "UTC"))
    )

    df = lf.collect()

    print(f"Time series: {df['lv_feeder_unique_id'].n_unique():,}")

    # Drop null feeders and wrong measurments
    bad_feeders = (
        df.filter(
            pl.col("mean_consumption_per_meter").is_null()
            | (pl.col("mean_consumption_per_meter") > 20000)
            | pl.col("geometry").struct.field("x").is_nan()
        )
        .select("lv_feeder_unique_id")
        .unique()
    )
    df = df.join(bad_feeders, on="lv_feeder_unique_id", how="anti")

    print(f"Dropped time series: {bad_feeders.height}")

    df = df.sort(["lv_feeder_unique_id", "data_collection_log_timestamp"])

    output_dir = Path("dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    unique_times = df.select(pl.col("data_collection_log_timestamp").unique().sort())
    t_val = unique_times.select(
        pl.col("data_collection_log_timestamp").quantile(0.80).dt.truncate("1d")
    ).item()
    t_test = unique_times.select(
        pl.col("data_collection_log_timestamp").quantile(0.90).dt.truncate("1d")
    ).item()

    buffer = timedelta(days=7)

    train_df = df.filter(pl.col("data_collection_log_timestamp") < t_val)
    valid_df = df.filter(
        (pl.col("data_collection_log_timestamp") >= (t_val - buffer))
        & (pl.col("data_collection_log_timestamp") < t_test)
    )
    test_df = df.filter(pl.col("data_collection_log_timestamp") >= (t_test - buffer))

    train_df.write_parquet(output_dir / "train.parquet")
    valid_df.write_parquet(output_dir / "val.parquet")
    test_df.write_parquet(output_dir / "test.parquet")


if __name__ == "__main__":
    main()
