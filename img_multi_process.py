from tkinter import SEPARATOR
from hexa_img import hexa_img
from pathlib import Path
import csv

if __name__ == "__main__":

    METAPATH = "hexa_meta.json"
    CSVFILE = "output.csv"
    SEPARATOR = "-"
    IMGFILE_DIR = "/media/hexaburbach/onetb/image_sync_manual/shoot1_lettuce"
    CONFIG = "/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/fcn_unet_s5-d16_128x128_320k_LeafDataset_T17.py"
    CHECKPOINT = "/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/iter_320000.pth"

    img_ext = (".jpg", ".JPG", ".png", ".PNG")
    areas = []

    imgs = sorted(filter(lambda path: path.suffix in img_ext, Path(
        IMGFILE_DIR).glob("*")), key=lambda path: str(path))

    hexa = hexa_img()
    """ mount segmentation model """
    hexa.mount(config_file=CONFIG, checkpoint_file=CHECKPOINT)

    """ process images """
    for img in imgs:
        hexa.load_img(filepath=img, metapath=METAPATH, separator=SEPARATOR)
        hexa.undistort().segment_with_model(show=True, pallete_path="output").compute_area().document(areas)

    """ Save areas """
    with open(CSVFILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(areas)