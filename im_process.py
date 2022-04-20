from tkinter import SEPARATOR
from hexa_img import hexa_img
from pathlib import Path
import csv
from dataclasses import replace
import argparse
from loguru import logger

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
        description="Get Camera Calibration Parameters")

    parser.add_argument("--img_dir",
                        # default="/media/hexaburbach/onetb/image_sync_manual/shoot1_lettuce",
                        default="G8T1-9400-0452-1S6N-1649786403.jpg",
                        help="Location of raw images' directory or an image file.")

    parser.add_argument("--meta",
                        default="hexa_meta.json",
                        help="Location of meta file.")

    parser.add_argument("--config",
                        default="/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/fcn_unet_s5-d16_128x128_320k_LeafDataset_T17.py",
                        help="Location of segmentation config file")

    parser.add_argument("--weight",
                        default="/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/iter_320000.pth",
                        help="Location of segmentation weight file")

    parser.add_argument("--csv",
                        default="output.csv",
                        help="Location of output to save csv file")

    parser.add_argument("--out",
                        default="output",
                        help="Location of output to save segmented images")

    parser.add_argument("--separator",
                        default="-",
                        help="Separation key word")

            

    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    METAPATH = args.meta
    CSVFILE = args.csv
    SEPARATOR = args.separator
    IMGFILE_DIR = args.img_dir
    CONFIG = args.config
    CHECKPOINT = args.weight
    OUTIMG = args.out

    img_ext = (".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG")
    areas = [["file_name","area_cm2", "volume_cm3"]]
    count_plants = 0

    img_path = Path(IMGFILE_DIR)
    if img_path.is_dir():
        imgs = sorted(filter(lambda path: path.suffix in img_ext, img_path.glob("*")), key=lambda path: str(path))

    elif img_path.is_file():
        if img_path.suffix in img_ext:
            imgs = [img_path.__str__()]
        else:
            logger.warning(f"Wrong suffix! Current suffix is {img_path.suffix}, not in {img_ext}.")
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
        hexa.undistort().segment_with_model(show=True, pallete_path=OUTIMG).compute_area().document(areas)
        count_plants = hexa.count

    """ Save areas """
    with open(CSVFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(areas)
    