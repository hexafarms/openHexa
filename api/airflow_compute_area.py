"""
Compute growth area for API.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V1.0
"""
import argparse
import os
import sys

from im_process import compute_area_api

sys.path.append(os.getcwd())


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
    
    area = compute_area_api(images)
    
    print({"input_args: ": args.image})
    print({"return value: ": area})
