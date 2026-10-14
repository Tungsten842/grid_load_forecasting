from datetime import date
from pathlib import Path

import boto3
import httpx
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm


def fetch_weave(output_dir):
    s3 = boto3.client(
        "s3",
        region_name="eu-west-2",
        config=Config(signature_version=UNSIGNED),
    )
    save_dir = output_dir / "weave"
    save_dir.mkdir(parents=True, exist_ok=True)

    bucket = "weave.energy"

    res = s3.list_objects_v2(Bucket=bucket, Prefix="smart-meter/")

    files = [f"{bucket}/{obj['Key']}" for obj in res["Contents"]]

    for s3_path in tqdm(files):
        bucket, key = s3_path.split("/", 1)
        filename = Path(key).name
        dest_path = save_dir / filename

        s3.download_file(bucket, key, str(dest_path))


def fetch_meteostat(
    output_dir,
    start_year=2024,
):
    save_dir = output_dir / "meteostat"
    save_dir.mkdir(parents=True, exist_ok=True)

    with httpx.Client(timeout=None) as client:
        for year in tqdm(range(start_year, date.today().year + 1)):
            file_path = save_dir / f"{year}.parquet"
            file_path.write_bytes(
                client.get(f"https://data.meteostat.net/hourly/{year}.parquet")
                .raise_for_status()
                .content
            )

        (save_dir / "stations.db").write_bytes(
            client.get("https://data.meteostat.net/stations.db")
            .raise_for_status()
            .content
        )


def main():
    output_dir = Path("./original_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    fetch_weave(output_dir)
    fetch_meteostat(output_dir)


if __name__ == "__main__":
    main()
