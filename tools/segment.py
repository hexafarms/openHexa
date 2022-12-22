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


def segment(
    images: Union[List[str], str],
    version: int,
    IMGFILE_DIR: str,
    mode: str = "mmseg",
    filter: bool = True,
) -> str:
    """Compute area for RESTapi."""

    assert mode in ["mmseg", "mmdet"], f"Mode is unknwon. Given value: {mode}"

    CONFIG = f"/openHexa/weights/{mode}/v{version}/config.py"
    CHECKPOINT = f"/openHexa/weights/{mode}/v{version}/weights.pth"

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
        hexa.segment_with_model(show=False, pallete_path=None, filter=filter)

        if hexa_base.pallete is None:
            hexa_base.pallete = [hexa.pallete.copy()]
        else:
            hexa_base.pallete.append(hexa.pallete.copy())

    return hexa_base.exportPallete()


if __name__ == "__main__":

    images = [
        "ecf_G8T1-K001-2173-0CVH-ir-1669676400.jpg",
        "ecf_G8T1-K001-2173-0CVH-ir-1669690800.jpg",
        "ecf_G8T1-K001-2173-0CVH-rgb-1669330800.jpg",
        "ecf_G8T1-K001-2173-0CVH-rgb-1669503600.jpg",
        "ecf_G8T1-K001-2173-0CVH-rgb-1670886000.jpg",
        "ecf_G8T1-K001-2173-0CVH-rgb-1670986800.jpg",
    ]
    METAPATH = "/home/huijo/codes/hexa_img_meta/data/meta/hexa_meta.json"
    IMGFILE_DIR = "/home/huijo/Downloads/brightCheck"
    mode = "mmdet"

    segment(images=images, version=0, IMGFILE_DIR=IMGFILE_DIR, mode=mode)
