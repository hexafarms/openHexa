from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File, Query
from predict import segment_api as segment
from fastapi.middleware.cors import CORSMiddleware
import os
import io
import re
from pathlib import Path
from openHexa.utils.helpers import getNewVersion

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

    weightDir = "/weights/mmdet"
    newVersion = getNewVersion(weightDir)

    config_file = os.path.join(weightDir, f"v{str(newVersion)}", "config.py")
    checkpoint_file = os.path.join(weightDir, f"v{str(newVersion)}", "weights.pth")

    imgDir = os.path.join("/images", location)
    Path(imgDir).mkdir(parents=True, exist_ok=True)

    # TODO: get files names from S3
    # TODO: get file names at the location from RDS (uhm... should I include RDS here???)
    # TODO: S3 - RDS => Download to local
    # TODO: Compute areas of files, and then update to RDS (or return to Airflow..??? is it safe??)
    # Maybe in this repo, only read permission to DB is awarded, and send the area info to Airflow? Then it can be micro-controlled, and safe!.
