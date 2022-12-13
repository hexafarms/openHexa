from pathlib import Path
import re
import os
from typing import List, Dict


def isCameraInCodes(file: Path, codes: List[str]) -> bool:
    """Is image file's camera code in the list of codes?"""
    if extractCode(file.stem) in codes:
        return True
    else:
        return False


def extractCode(file_name: str) -> str:
    """Extract camera code from file name."""
    camRegex = re.compile(
        r"\w\w\w\w-\w\w\w\w-\w\w\w\w-\w\w\w\w"
    )  # Match the Code Pattern.
    return camRegex.search(file_name)[0]


def sortFilesByTime(files: List[str]) -> List[List[str]]:
    files = sorted(files, key=lambda x: int(x.stem.split("-")[-1]))
    FilesByTimeWithList = []
    epochTime = 0
    for file in files:
        epochTimeOfFile = int(file.stem.split("-")[-1])
        if epochTimeOfFile != epochTime:
            FilesByTimeWithList.append([file])  # new time, then add a new list.
            epochTime = epochTimeOfFile
        else:
            FilesByTimeWithList[-1].append(
                file
            )  # same time, then add to the recent list
    return FilesByTimeWithList


def sortFilesByColor(files: List[str]) -> Dict[str, List[str]]:
    """Only rgb and ir color are valid."""

    FilesByColor = {"rgb": [], "ir": [], "codes": []}
    for file in files:
        colorType = file.stem.split("-")[-2]
        assert colorType in [
            "rgb",
            "ir",
        ], f"Invalid color type. Check the file name format. Given color type is {colorType}."
        FilesByColor[colorType].append(file)

        code = extractCode(file.stem)
        if code not in FilesByColor.get("codes"):
            FilesByColor["codes"].append(code)

    if len(FilesByColor["rgb"]) % 2 == 0 and len(FilesByColor["ir"]) % 2 == 0:
        return FilesByColor
    else:
        return None


def getNewVersion(DIR: str):
    """Extract the newest version out of the list of versions."""
    new_version = 0
    versions = [os.path.basename(x[0]) for x in os.walk(DIR)][
        1:
    ]  # exclude the parent path

    if version is None:
        "Find the best version if not given"

        if len(versions) == 0:
            NameError("No proper version inside weight folder!")

        for v in versions:
            version = int(re.search("v(.*)", v).group(1))
            if version > new_version:
                new_version = version

    else:
        new_version = version[1:]
    return new_version
