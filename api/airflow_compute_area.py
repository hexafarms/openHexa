import sys
import argparse

def parse_args():
    '''Parse input arguments'''
    parser = argparse.ArgumentParser(
                description="Compute area of plants")
    parser.add_argument("image", 
                        nargs='+', 
                        type=str, 
                        help="images to be processed.")
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print({"input_args: ": args.image })