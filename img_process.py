from sys import meta_path
from tkinter import SEPARATOR
from hexa_img import hexa_img

if __name__ == "__main__":

    METAPATH = "hexa_meta.json"
    SEPARATOR = "-"
    IMGFILE = "images/dev-0-1639483201.jpg"
    CONFIG = "/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/fcn_unet_s5-d16_128x128_320k_LeafDataset_T17.py"
    CHECKPOINT = "/home/hexaburbach/codes/mmsegmentation/fast_api/best_model/iter_320000.pth"


    hexa = hexa_img()
    hexa.load_img(filepath = IMGFILE, metapath=METAPATH, separator = SEPARATOR)
    hexa.undistort().segment(config_file=CONFIG,checkpoint_file=CHECKPOINT, show = True, pallete_path=".").compute_area()

