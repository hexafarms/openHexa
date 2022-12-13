import boto3
import re
from loguru import logger
import os
from pathlib import Path
from typing import Dict


def _connect_aws_s3(AWS_KEY: str, AWS_SECRET: str, AWS_REGION: str):
    """Connect to aws client"""

    # Get credentials to access to S3 bucket.
    AWS_KEY = os.environ["AWS_KEY"]
    AWS_SECRET = os.environ["AWS_SECRET"]
    AWS_REGION = os.environ["AWS_REGION"]

    s3_client = boto3.client(
        "s3",
        region_name=AWS_REGION,
        aws_access_key_id=AWS_KEY,
        aws_secret_access_key=AWS_SECRET,
    )
    return s3_client


def _check_bucket(s3_client, bucket_name):
    """Check if the bucket exist in aws S3"""
    bucket = None
    for i in s3_client.list_buckets()["Buckets"]:
        if i["Name"] == bucket_name:
            bucket = bucket_name
            break
    if bucket == None:
        raise NameError("AWS S3 bucket is not found.")

    return bucket


def prepare_configs(cv_mode: str):
    """Prepare configs for segmentation."""

    assert cv_mode in ["mmseg", "mmdet"], f"Mode is inappropriate. Input: {cv_mode}"

    s3_client = _connect_aws_s3()

    "Check the bucket is inside AWS S3"
    blink_bucket_name = _check_bucket(s3_client, "configs-and-weights")
    meta_bucket_name = _check_bucket(s3_client, "hexameta")

    """Find the newest version"""
    relevant_versions = []

    new_version = 0
    for j in s3_client.list_objects(Bucket=blink_bucket_name)["Contents"]:
        if j["Key"].split("/")[0] == cv_mode:
            relevant_versions.append(j)
            if re.search("\/v(.*)\/", j["Key"], 0) is not None:
                version = int(re.search("\/v(.*)\/", j["Key"], 0).group(1))
                if version > new_version:
                    new_version = version

    up2date_config_v = f"v{new_version}"

    """Check if configs and weights are in volumes"""
    config_file_path = os.path.join("/weights", cv_mode, up2date_config_v, "config.py")
    weights_path = os.path.join("/weights", cv_mode, up2date_config_v, "weights.pth")
    Path(config_file_path).parents[0].mkdir(parents=True, exist_ok=True)

    meta_path = os.path.join("/meta", "hexa_meta.json")
    Path(meta_path).parents[0].mkdir(parents=True, exist_ok=True)

    """MetaData is downloaded everytime."""
    aws_meta_path = os.path.join("camera", "hexa_meta.json")
    s3_client.download_file(meta_bucket_name, aws_meta_path, str(meta_path))
    logger.info("Meta file for image segmentation is downloaded.")

    """Download if not the new version exist."""
    if not os.path.exists(config_file_path):
        aws_config_path = os.path.join(cv_mode, up2date_config_v, "config.py")
        s3_client.download_file(
            blink_bucket_name, aws_config_path, str(config_file_path)
        )
        logger.info("Config file for image segmentation is downloaded.")

    if not os.path.exists(weights_path):
        aws_weights_path = os.path.join(cv_mode, up2date_config_v, "weights.pth")
        s3_client.download_file(blink_bucket_name, aws_weights_path, str(weights_path))
        logger.info("Weights file for image segmentation is downloaded.")

    return s3_client
