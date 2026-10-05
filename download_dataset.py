from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.client import Config
from tqdm import tqdm


def main():
    output_dir = Path("./original_dataset")
    output_dir.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client(
        "s3",
        region_name="eu-west-2",
        config=Config(signature_version=UNSIGNED),
    )

    bucket = "weave.energy"

    res = s3.list_objects_v2(Bucket=bucket, Prefix="smart-meter/")

    files = [f"{bucket}/{obj['Key']}" for obj in res["Contents"]]

    for s3_path in tqdm(files):
        bucket, key = s3_path.split("/", 1)
        filename = Path(key).name
        dest_path = output_dir / filename

        s3.download_file(bucket, key, str(dest_path))


if __name__ == "__main__":
    main()
