"""
Examples of running Hexa img module.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V1.0
"""
import argparse
import os
from dataclasses import replace
from pathlib import Path
from typing import List, Union
import numpy as np
import cv2

from loguru import logger
from tqdm import tqdm
import torch
import gc

from openHexa.imgInstance import hexa_img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description="Get Camera Calibration Parameters")

    parser.add_argument(
        "--img_dir",
        default="images",
        help="Location of raw images' directory or an image file.",
    )

    parser.add_argument(
        "--meta", default="hexa_meta.json", help="Location of meta file."
    )

    parser.add_argument(
        "--config",
        default="/home/hexaburbach/codes/mmsegmentation/fast_api/\
best_model/fcn_unet_s5-d16_128x128_320k_LeafDataset_T17.py",
        help="Location of segmentation config file",
    )

    parser.add_argument(
        "--weight",
        default="/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/iter_320000.pth",
        help="Location of segmentation weight file",
    )

    parser.add_argument(
        "--csv", default="output.csv", help="Location of output to save csv file"
    )

    parser.add_argument(
        "--out", default="output", help="Location of output to save segmented images"
    )

    parser.add_argument("--separator", default="-", help="Separation key word")

    parser.add_argument("--remove", default=False,
                        help="remove irrelvant reigion.")

    args = parser.parse_args()
    return args


def compute_area_api(
    images: List[str],
    version: int,
    METAPATH: str,
    IMGFILE_DIR: str,
    mode: str = "mmseg",
) -> str:
    """Compute area for RESTapi."""

    assert mode in ["mmseg", "mmdet"], f"Mode is unknwon. Given value: {mode}"

    CONFIG = f"/weights/{mode}/v{version}/config.py"
    CHECKPOINT = f"/weights/{mode}/v{version}/weights.pth"

    output = {}

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT, mode=mode)

    """ process images """
    for img in tqdm(images):
        img_full_path = os.path.join(IMGFILE_DIR, img)
        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path, metapath=METAPATH)
        hexa.undistort().segment_with_model(
            show=False, pallete_path=None
        ).compute_area().document(output)

    if torch.cuda.is_available():
        # free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return output


def compute_raw_area_api(
    images: Union[List[str], str],
    version: int,
    IMGFILE_DIR: str,
    mode: str = "mmseg",
) -> str:
    """Compute area for RESTapi. It returns with meta data to save into DB and return to internal service such as hexaBM"""

    assert mode in ["mmseg", "mmdet"], f"Mode is unknwon. Given value: {mode}"

    CONFIG = f"/openHexa/weights/{mode}/v{version}/config.py"
    CHECKPOINT = f"/openHexa/weights/{mode}/v{version}/weights.pth"

    # for debugging locally.
    # CONFIG = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/config.py"
    # CHECKPOINT = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/weights.pth"

    output = {}

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT, mode=mode)

    if isinstance(images, str):
        # if single image is feeded, then make it as a list.
        images = [images]

    """ process images """
    for img in tqdm(images):
        img_full_path = os.path.join(IMGFILE_DIR, img)
        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path).measureBright()
        hexa.segment_with_model(show=False, pallete_path=None).compute_area().document(
            output
        )
    if torch.cuda.is_available():
        # free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return output

def classify_health_api(
    images: Union[List[str], str],
    version: int,
    IMGFILE_DIR: str,
    mode: str = "mmcls",
) -> str:
    """Compute area for RESTapi. It returns with meta data to save into DB and return to internal service such as hexaBM"""

    assert mode in ["mmcls"], f"Mode is unknwon. Given value: {mode}"

    CONFIG = f"/openHexa/weights/{mode}/v{version}/config.py"
    CHECKPOINT = f"/openHexa/weights/{mode}/v{version}/weights.pth"

    # for debugging locally.
    # CONFIG = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/config.py"
    # CHECKPOINT = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/weights.pth"

    output = {}

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT, mode=mode)

    if isinstance(images, str):
        # if single image is feeded, then make it as a list.
        images = [images]

    """ process images """
    for img in tqdm(images):
        img_full_path = os.path.join(IMGFILE_DIR, img)
        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path).measureBright()
        hexa.segment_with_model(show=False, pallete_path=None).document(
            output
        )
    if torch.cuda.is_available():
        # free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return output

def compute_area(args, include_header=False):
    """Compute area for python module."""
    METAPATH = args.meta
    SEPARATOR = args.separator
    IMGFILE_DIR = args.img_dir
    CONFIG = args.config
    CHECKPOINT = args.weight
    OUTIMG = args.out
    REMOVE = args.remove
    REMOVE = [
        [[400, 0], [0, 400]],
        [["end", 100], [1100, "end"]],
        [[150, "end"], [0, 550]],
    ]

    img_ext = (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG")
    if include_header:
        areas = [["file_name", "area_mm2", "volume_mm3"]]
    else:
        areas = []
    count_plants = 0

    img_path = Path(IMGFILE_DIR)
    if img_path.is_dir():
        imgs = sorted(
            filter(lambda path: path.suffix in img_ext, img_path.glob("*")),
            key=lambda path: str(path),
        )

    elif img_path.is_file():
        if img_path.suffix in img_ext:
            imgs = [img_path.__str__()]
        else:
            logger.warning(
                f"Wrong suffix! Current suffix is {img_path.suffix}, not in {img_ext}."
            )
    else:
        logger.warning("Wrong input format. Check your input argument.")

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT)

    """ process images """
    for img in imgs:
        hexa = replace(hexa_base)
        hexa.update_count(count_plants)
        hexa.load_img(filepath=img, metapath=METAPATH, separator=SEPARATOR)
        if REMOVE:
            hexa.remove(REMOVE)
        hexa.undistort().segment_with_model(
            show=True, pallete_path=OUTIMG
        ).compute_area().document(areas, graph=False)
        count_plants = hexa.count

    if torch.cuda.is_available():
        # free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return areas


def compute_ndvi(rgb: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """
    Compute NDVI from rgb and ir images.

    rgb: numpy array of rgb image (3 or 4 color channels)
    ir: numpy array of ir image (3 or 4 color channels)

    return: numpy array of ndvi (1 color channel)
    """
    r = rgb[...,0].astype(float)
    ir = cv2.cvtColor(ir, cv2.COLOR_RGB2GRAY).astype(float)

    ndvi = np.divide(ir-r, ir+r, out=np.zeros_like(ir), where=ir+r!=0)

    return ndvi

def bytes2array(data: bytes) -> np.ndarray:
    """
    Convert bytes to RGB array

    data: bytes

    return: numpy array RGB
    """

    npimg = np.frombuffer(data, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    return frame

if __name__ == "__main__":

    # images = [
    #     "ecf_G8T1-K001-2173-0CVH-ir-1669676400.jpg",
    #     "ecf_G8T1-K001-2173-0CVH-ir-1669690800.jpg",
    #     "ecf_G8T1-K001-2173-0CVH-rgb-1669330800.jpg",
    #     "ecf_G8T1-K001-2173-0CVH-rgb-1669503600.jpg",
    #     "ecf_G8T1-K001-2173-0CVH-rgb-1670886000.jpg",
    #     "ecf_G8T1-K001-2173-0CVH-rgb-1670986800.jpg",
    # ]
    METAPATH = "/home/huijo/codes/hexa_img_meta/data/meta/hexa_meta.json"
    IMGFILE_DIR = "/home/huijo/Pictures/species"
    import glob
    images = glob.glob(f"{IMGFILE_DIR}/*.jpg")
    mode = "mmdet"

    compute_raw_area_api(images=images, version=3,
                         IMGFILE_DIR=IMGFILE_DIR, mode=mode)

    # args = parse_args()
    # areas = compute_area(args, include_header=True)

    # """ Save areas to csv """
    # with open(args.csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(areas)
