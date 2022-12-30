import yaml
from pathlib import Path
from typing import List, Dict
from loguru import logger
from openHexa.utils.helpers import *
from openHexa.imgInstance import HexaPallelDepth


def sortFiles(LOCAL_PATH: str, setup: Dict) -> List[Dict[str, List[str]]]:
    """sort files based on camera and time.

    Args:
        LOCAL_PATH (str): Local path where image are located.
        setup (Dict): Dictionary from credential files

    Returns:
        List[Dict[str, List[Path]]]: {"rgb": [str], "ir": [str], "codes": [str]}
    """

    imgFiles = Path(LOCAL_PATH).glob("*.jpg")
    # Make nested list with the number of cameras.
    imgFilesPerLocation = [[]] * len(setup)

    for i, (key, val) in enumerate(setup.items()):
        logger.info(f"Image files of {key} are searched.")
        camera_codes = val.get("camera")

        for imgFile in imgFiles:
            if isCameraInCodes(imgFile, camera_codes):
                imgFilesPerLocation[i].append(imgFile)

    filesByTime = [sortFilesByTime(i) for i in imgFilesPerLocation]
    filesByColor = []

    for filesPerCodes in filesByTime:
        for filesPerTime in filesPerCodes:
            fileByColor = sortFilesByColor(filesPerTime)
            if fileByColor is None:
                continue
            else:
                filesByColor.append(fileByColor)
    return filesByColor


def findDistance(setup: Dict, codes: List) -> int:
    """extract distance between two camera using camera code

    Args:
        setup (Dict): Dictionary from credential files
        codes (List): two camera codes

    Returns:
        int: distance between two cameras
    """
    for _, val in setup.items():

        if sorted(val.get("camera")) == sorted(codes):
            return val.get("distance")

    return KeyError("Can't find distance from two cameras from the setup file.")


def findDefaultDepth(setup: Dict, codes: List) -> int:
    """Find default depth where the came is mounted

    Args:
        setup (Dict): Dictionary from credential files
        codes (List): two camera codes

    Returns:
        int: depth from cameras to the target.
    """
    for _, val in setup.items():

        if sorted(val.get("camera")) == sorted(codes):
            return val.get("depth")

    return KeyError("Can't find depth of cameras from the setup file.")


def main(LOCAL_PATH, setup):
    filesByColor = sortFiles(LOCAL_PATH, setup)
    depth_objects = []

    for fileByColor in filesByColor:

        cam_codes = fileByColor["codes"]

        depth_objects.append(
            HexaPallelDepth(
                distance=findDistance(setup, cam_codes),
                cam_codes=cam_codes,
                base_depth=findDefaultDepth(setup, cam_codes),
            )
            .registerRGB(fileByColor["rgb"])
            .registerIR(fileByColor["ir"])
            .computeDisparity()
        )


if __name__ == "__main__":
    # TODO: Build same pipeline using click library. (using system arguments.)

    LOCAL_PATH = "/media/huijo/SSD_Storage/S3_Download/ecf"

    with open("credential/hexa.yaml", "r") as stream:
        setup = yaml.safe_load(stream)

    main(LOCAL_PATH, setup)
