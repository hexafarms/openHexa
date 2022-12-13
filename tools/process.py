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
import re
from typing import List

from loguru import logger

from openHexa.imgInstance import hexa_img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Get Camera Calibration Parameters")

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

    parser.add_argument("--remove", default=False, help="remove irrelvant reigion.")

    args = parser.parse_args()
    return args


def compute_area_api(
    images: List[str],
    version: str = None,
    METAPATH: str = "/Hexa_image/meta/hexa_meta.json",
    IMGFILE_DIR: str = "/Hexa_image/data/images/pictures",
    mode: str = "mmseg",
) -> str:
    """Compute area for RESTapi."""

    SEPARATOR = "-"

    assert mode in ["mmseg", "mmdet"], f"Mode is unknwon. Given value: {mode}"

    if version is None:
        "Find the best version if not given"
        new_version = 0
        versions = [os.path.basename(x[0]) for x in os.walk("/weights")][
            1:
        ]  # exclude the parent path

        if len(versions) == 0:
            NameError("No proper version inside weight folder!")

        for v in versions:
            version = int(re.search("v(.*)", v).group(1))
            if version > new_version:
                new_version = version

    else:
        new_version = version[1:]

    CONFIG = f"/weights/{mode}/v{new_version}/config.py"
    CHECKPOINT = f"/weights/{mode}/v{new_version}/weights.pth"

    output = {}

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT, mode=mode)

    """ process images """
    for img in images:
        img_full_path = os.path.join(IMGFILE_DIR, img)
        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path, metapath=METAPATH, separator=SEPARATOR)
        hexa.undistort().segment_with_model(
            show=False, pallete_path=None
        ).compute_area().document(output)

    return output


def compute_area_raw_api(
    images: List,
    version: str = None,
    METAPATH: str = "/openHexa/meta/hexa_meta.json",
    IMGFILE_DIR: str = "/openHexa/data/images/pictures",
) -> str:
    """Compute area for RESTapi, output format is the raw list."""

    SEPARATOR = "-"
    # TODO get rid of meta file part (especially, pixel ratio part)

    new_version = 0
    versions = [os.path.basename(x[0]) for x in os.walk("/weights")][
        1:
    ]  # exclude the parent path

    if version is None:
        "Find the best version if not given"

        if len(versions) == 0:
            NameError("No proper version inside weight folder!")

        for v in versions:
            version = int(re.search("v(.*)", v).group(1))
            if version > new_version:
                new_version = version

    else:
        new_version = version[1:]

    CONFIG = f"/weights/v{new_version}/config.py"
    CHECKPOINT = f"/weights/v{new_version}/weights.pth"

    areas = []

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT)

    """ process images """
    for img in images:
        img_full_path = os.path.join(IMGFILE_DIR, img)
        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path, metapath=METAPATH, separator=SEPARATOR)
        hexa.undistort().segment_with_model(
            show=False, pallete_path=None
        ).compute_area().document(areas, graph=False, volume=False)

    return areas


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

    return areas


if __name__ == "__main__":

    images = [
        "ecf_G8T1-K001-2173-0FSR-1668034800.jpg",
    ]
    METAPATH = "/home/huijo/codes/hexa_img_meta/data/meta/hexa_meta.json"
    IMGFILE_DIR = "/home/huijo/Downloads"
    mode = "mmdet"

    compute_area_api(
        images=images, METAPATH=METAPATH, IMGFILE_DIR=IMGFILE_DIR, mode=mode
    )

    # args = parse_args()
    # areas = compute_area(args, include_header=True)

    # """ Save areas to csv """
    # with open(args.csv, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerows(areas)