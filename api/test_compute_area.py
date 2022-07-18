"""
Compute growth area for API.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V1.0
"""
import argparse
import os
import sys

sys.path.append(os.getcwd())
from im_process import compute_area_api


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description="Compute area of plants")
    parser.add_argument("image",
                        nargs='+',
                        type=str,
                        help="images to be processed.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    images = args.image
    meta_path = "hexa_meta.json"
    img_path = "demo/undistort_demo.jpg"
    
    area = compute_area_api(images, METAPATH=meta_path, IMGFILE_DIR=img_path)
    if area > 0:
        print("Test is Successful!.")
    
