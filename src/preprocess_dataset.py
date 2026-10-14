import os
import sqlite3
from datetime import timedelta
from pathlib import Path

from scipy.spatial import cKDTree

os.environ["POLARS_UNKNOWN_EXTENSION_TYPE_BEHAVIOR"] = "load_as_storage"

import polars as pl


def get_weather(df):
    folder = Path("original_dataset/meteostat")
    files = str(folder / "*.parquet")
    timestamp = "data_collection_log_timestamp"

    # Find stations that have weather data
    with sqlite3.connect(folder / "stations.db") as conn:
        stations = pl.read_database(
            """
            SELECT id AS station, latitude, longitude
            FROM stations
            WHERE latitude IS NOT NULL AND longitude IS NOT NULL
            """,
            conn,
        )

    available = pl.scan_parquet(files).select("station").unique().collect()

    stations = available.join(stations, on="station").unique(subset=["station"])

    # Assign the nearest weather station to each location
    points = df.select("longitude", "latitude").unique().drop_nulls()

    tree = cKDTree(stations.select("longitude", "latitude").to_numpy())
    _, indices = tree.query(points.to_numpy())

    location_map = points.with_columns(
        stations["station"].gather(indices).alias("station")
    )

    result = df.join(
        location_map,
        on=["longitude", "latitude"],
        how="left",
    )

    # Load weather data for the required stations and dates
    target_stations = location_map["station"].unique().to_list()

    weather = (
        pl.scan_parquet(files)
        .filter(pl.col("station").is_in(target_stations))
        .select(
            "station",
            "time",
            pl.col("temp").alias("temperature"),
            pl.col("rhum").alias("humidity"),
        )
        .collect()
    )

    # Convert weather observations to 30-minute intervals
    weather = (
        weather.with_columns(pl.col("time").cast(df[timestamp].dtype).alias(timestamp))
        .drop("time")
        .sort(["station", timestamp])
        .upsample(
            time_column=timestamp,
            every="30m",
            group_by="station",
        )
        .with_columns(
            pl.col("temperature").interpolate().over("station"),
            pl.col("humidity").interpolate().over("station"),
        )
    )

    # Attach weather data to the original rows
    return result.join(weather, on=["station", timestamp]).drop(
        ["station", "latitude", "longitude"]
    )


def main():
    directory = Path("original_dataset")

    columns = [
        "lv_feeder_unique_id",
        "total_consumption_active_import",
        "aggregated_device_count_active",
        "data_collection_log_timestamp",
        "geometry",
    ]
    lf = pl.scan_parquet(directory / "weave" / "*.parquet").select(columns)

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

    lf = lf.with_columns(
        pl.col("geometry").ext.storage().struct.field("x").alias("longitude"),
        pl.col("geometry").ext.storage().struct.field("y").alias("latitude"),
    ).drop("geometry")

    df = lf.collect()

    print(f"Time series: {df['lv_feeder_unique_id'].n_unique():,}")

    # Drop null feeders and wrong measurments
    bad_feeders = (
        df.filter(
            pl.col("mean_consumption_per_meter").is_null()
            | (pl.col("mean_consumption_per_meter") > 20000)
            | pl.col("latitude").is_nan()
        )
        .select("lv_feeder_unique_id")
        .unique()
    )
    df = df.join(bad_feeders, on="lv_feeder_unique_id", how="anti")

    print(f"Dropped time series: {bad_feeders.height}")

    df = df.sort(["lv_feeder_unique_id", "data_collection_log_timestamp"])

    df = get_weather(df)

    print(df.columns)

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
