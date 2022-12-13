"""
Compute meta data.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V1.1
"""
from openHexa.imgInstance import hexa_process


def getMeta(
    Input: str,
    PatchChecker: str,
    Separator: str,
    MetaPath: str,
    CornerWidth: int,
    CornerHeight: int,
    ActualDim: int,
):
    """Compute Meta file from check image and Hough circle."""
    hexa = hexa_process()
    hexa.calibrate(
        imgpath_checker=PATHCHECKER,
        corner_w=CORNER_WIDTH,
        corner_h=CORNER_HEIGHT,
        metafile=METAPATH,
        separator=SEPARATOR,
    )

    hexa.compute_px_ratio(
        filepath=INPUT, metapath=METAPATH, separator=SEPARATOR, actual_dim=ACTUAL_DIM
    )


if __name__ == "__main__":

    INPUT = "checker/burbach-left-16569425.jpg"  # for px_ratio
    PATHCHECKER = "checker"  # for calibration
    SEPARATOR = "-"
    METAPATH = "."
    CORNER_WIDTH = 9
    CORNER_HEIGHT = 6
    ACTUAL_DIM = 50  # MM

    getMeta(
        INPUT, PATHCHECKER, SEPARATOR, METAPATH, CORNER_WIDTH, CORNER_HEIGHT, ACTUAL_DIM
    )
