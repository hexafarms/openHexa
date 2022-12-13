from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from fastapi.middleware.cors import CORSMiddleware
import boto3
import os
from pathlib import Path
from openHexa.utils.helpers import getNewVersion
from tools.process import compute_area_api
from configs.aws import prepare_configs

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hexafarms": "openHexa V1"}


@app.get("/instantseg")
async def sync_instantSeg(location: str):

    if location is None:
        return {"Warning": "Location is not provided!"}

    mode = "mmdet"

    weightDir = os.path.join("/weights", mode)

    # Download config, weight, meta file.
    s3_client=prepare_configs(mode)
    
    newVersion = getNewVersion(weightDir)

    config_file = os.path.join(weightDir, f"v{str(newVersion)}", "config.py")
    checkpoint_file = os.path.join(weightDir, f"v{str(newVersion)}", "weights.pth")

    imgDir = os.path.join("/images", location)
    Path(imgDir).mkdir(parents=True, exist_ok=True)

    imgsInS3 = set([i['Key'] for i in s3_client.list_objects(Bucket=location)['Contents']])

    # TODO: get files names from S3
    # TODO: get file names at the location from RDS (uhm... should I include RDS here???)
    # TODO: S3 - RDS - local => Download to local
    # TODO: Compute areas of files, and then update to RDS (or return to Airflow..??? is it safe??)
    # Maybe in this repo, only read permission to DB is awarded, and send the area info to Airflow? Then it can be micro-controlled, and safe!.


    imgs = [] # should be computed

    areas = compute_area_api(
        imgs, newVersion, METAPATH = "/meta/hexa_meta.json", IMGFILE_DIR= imgDir, mode= mode)
    
    return ORJSONResponse(areas)
    