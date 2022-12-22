from fastapi import FastAPI, UploadFile, File
from fastapi.responses import ORJSONResponse, Response, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import io
import shutil
from PIL import Image
import os
from pathlib import Path
from openHexa.utils.helpers import getNewVersion
from tools.process import compute_raw_area_api
from tools.segment import segment
from configs.aws import prepare_configs, getFiles
import glob
from typing import Union, List
import imageio.v3 as iio

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
    return {"openHexa": "V1"}

@app.post("/instantsegShow")
async def show_instantSeg(file: UploadFile = File(...), version: Union[str, None] = None):

    mode = "mmdet"

    weightDir = os.path.join("/openHexa/weights", mode)

    # Download config, weight, meta file, and return S3 client.
    
    newVersion = getNewVersion(weightDir, version)


    imgDir = file.filename

    with open(imgDir, "wb+") as file_object:
        file_object.write(file.file.read())

    pallete = segment(
        imgDir, newVersion, IMGFILE_DIR= "/openHexa", mode= mode)

    bytes_image = io.BytesIO()
    im = Image.fromarray(pallete)
    im.save(bytes_image, format="PNG")
    
    return Response(content=bytes_image.getvalue(), headers={'processDone':file.filename}, media_type=("image/jpeg"or"image/png"or"image/jpg"))

@app.post("/instantsegsGif")
async def gif_instantSeg(files: List[UploadFile] = File(...), version: Union[str, None] = None):

    mode = "mmdet"

    weightDir = os.path.join("/openHexa/weights", mode)

    # Download config, weight, meta file, and return S3 client.
    
    newVersion = getNewVersion(weightDir, version)

    imgDir = []

    for file in files:

        with open(file.filename, "wb+") as buffer:
            shutil.copyfileobj(file.file, buffer)

        imgDir.append(file.filename)

    pallete = segment(
        imgDir, newVersion, IMGFILE_DIR= "/openHexa", mode= mode)

    imgs = [Image.fromarray(i) for i in pallete]

    bytes_image = io.BytesIO()
    iio.imwrite(bytes_image, imgs, plugin="pillow", extension= ".gif", duration=1000, loop=0)
    bytes_image.seek(0)
        
    return StreamingResponse(bytes_image, media_type='image/gif')
   

@app.get("/instantseg")
async def sync_instantSeg(location: str, version: Union[str, None] = None):

    if location is None:
        return {"Warning": "Location is not provided!"}

    mode = "mmdet"

    weightDir = os.path.join("/openHexa/weights", mode)

    # Download config, weight, meta file, and return S3 client.
    
    s3_client=prepare_configs(mode, weightDir)
    
    newVersion = getNewVersion(weightDir, version)

    imgDir = os.path.join("/openHexa/images", location)
    Path(imgDir).mkdir(parents=True, exist_ok=True)

    bucketName = 'blink-'+location

    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucketName, Prefix=f"{location}")
    imgsS3 = set([
        obj['Key'] for page in pages for obj in page['Contents']  
        if ('ir' not in obj['Key'])
    ]) # exclude ir images.
    # imgsS3 = set([i['Key'] for i in s3_client.list_objects(Bucket=bucketName)['Contents'] if "ir" not in i['Key']]) 
    imgsDB = set(getFiles(
        sql= f"SELECT top_view.file_name, img_format.format FROM top_view \
            JOIN img_format \
            ON img_format.img_format_id = top_view.img_format \
            WHERE top_view.location = \
            (SELECT location_id from locations WHERE location='{location}');"
        ) )
    imgsLocal = set(glob.glob(os.path.join(imgDir, '*.jpg'))) | set(glob.glob(os.path.join(imgDir, '*.png')))

    imgs2Download = list(imgsS3 - imgsDB - imgsLocal)

    for file in imgs2Download:
        s3_client.download_file(bucketName, file, os.path.join(imgDir, file))

    imgs2Update = list(imgsLocal - imgsDB)

    areas = compute_raw_area_api(
        imgs2Update, newVersion, IMGFILE_DIR= imgDir, mode= mode)
    
    return ORJSONResponse(areas)
    