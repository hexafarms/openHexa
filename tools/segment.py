"""
Examples of running Hexa img module.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V1.1
"""
import os
from dataclasses import replace
from typing import List, Union
from openHexa.imgInstance import hexa_img
import torch
import gc


def segment(
    images: Union[List[str], str],
    version: int,
    IMGFILE_DIR: str,
    mode: str = "mmseg",
    filter: bool = False,
    pallete_path: str = None,
    show: bool = False
) -> str:
    """Compute area for RESTapi."""

    assert mode in ["mmseg", "mmdet"], f"Mode is unknwon. Given value: {mode}"

    CONFIG = f"/openHexa/weights/{mode}/v{version}/config.py"
    CHECKPOINT = f"/openHexa/weights/{mode}/v{version}/weights.pth"

    # for debugging locally.
    # CONFIG = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/config.py"
    # CHECKPOINT = f"/home/huijo/Desktop/mnt/weights/{mode}/v{version}/weights.pth"

    hexa_base = hexa_img()
    """ mount segmentation model """
    hexa_base.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT, mode=mode)

    if isinstance(images, str):
        # if single image is feeded, then make it as a list.
        images = [images]

    """ process images """
    for img in images:
        img_full_path = os.path.join(IMGFILE_DIR, img)

        hexa = replace(hexa_base)
        hexa.load_img(filepath=img_full_path)
        hexa.segment_with_model(
            show=show, pallete_path=pallete_path, filter=filter)

        if hexa_base.pallete is None:
            hexa_base.pallete = [hexa.pallete.copy()]
        else:
            hexa_base.pallete.append(hexa.pallete.copy())

    if torch.cuda.is_available():
        # free GPU memory!
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return hexa_base.exportPallete()


if __name__ == "__main__":

    images = [
        "G8T1-K001-2173-0FUF-rgb-1668162570.jpg",
        "G8T1-K001-2173-0FR1-rgb-1668351600.jpg"
    ]
    METAPATH = "/home/huijo/codes/hexa_img_meta/data/meta/hexa_meta.json"
    IMGFILE_DIR = "/home/huijo/Pictures/demo"
    mode = "mmdet"

    segment(images=images, version=3, IMGFILE_DIR=IMGFILE_DIR, mode=mode,
            pallete_path=None, show=False)
