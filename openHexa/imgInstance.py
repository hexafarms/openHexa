"""
Hexa img module.

Author: Huijo Kim
Email: huijo.k@hexafarms
Version: V2.0
"""
import json
import os
import re
import statistics
import math
from pathlib import Path

# This is for airflow api to connect into mmseg
from dataclasses import dataclass
from json.decoder import JSONDecodeError
from typing import Any, Dict, List, Optional, Tuple, Union
from time import gmtime, strftime

# external library for image processing.
import cv2
import numpy as np
import torch
from typing import List
from matplotlib import pyplot as plt

# logging module
from loguru import logger


@dataclass
class hexa_img:
    """Image format handling Computer Vision task."""

    img: np.ndarray = None
    mask: Union[np.ndarray, List[np.ndarray]] = None
    pallete: Union[np.ndarray, List[np.ndarray]] = None
    name: str = None
    param: Optional[Dict[str, int]] = None  # camera parameter
    # mm2 per pixel, 0.3 is average at 40cm height
    ratio: Optional[float] = 0.3
    area: Optional[float] = 0
    volume: Optional[float] = 0
    count: int = 1  # the number of plants in the bench
    model: Any = None
    cv_mode: str = None
    bright: int = None
    bbox: List = None
    health: Dict = None
    maskRatio: float = None

    def load_img(self, filepath: str, metapath=None):
        """Load image."""
        self.img = cv2.imread(str(filepath))
        assert type(self.img) != type(None), f"no file {filepath} exist!"

        self.name = os.path.basename(filepath)

        if metapath is None:
            logger.info(
                "It is the raw-mode. No camera undistortion and adjust area.")
            return self

        else:
            # load meta data
            with open(metapath, "r+") as j:
                try:
                    data = json.load(j)
                except JSONDecodeError:
                    logger.warning(
                        "Json file should not be empty. Delete the empty file and run again."
                    )
                    return 0
            # check if the value is inside meta.
            camRegex = re.compile(r"\w\w\w\w-\w\w\w\w-\w\w\w\w-\w\w\w\w")
            camera_code = camRegex.search(filepath)[0]

            if camera_code not in data.keys():
                logger.warning("no camera info in meta data.")
                return self

            if "parameters" in data[camera_code].keys():
                logger.success(f"parameters of {filepath} is loaded.")
                self.param = data[camera_code]["parameters"]
            else:
                logger.warning(
                    f"parameters of {filepath} don't exist in {metapath}. no undistortion will be applied."
                )

            if "pixel2mm" in data[camera_code].keys():
                logger.success(
                    f"ratio of pixel to mm2 of {filepath} is loaded.")
                self.ratio = data[camera_code]["pixel2mm"]
            else:
                logger.warning(
                    f"ratio of pixel to mm2 of {filepath} don't exist in {metapath}. Area will be in 0.3 mm^2 per pixel."
                )
            return self

    def remove(self, points: List):
        """Remove of parts of image."""
        h, w, _ = self.shape
        points_arr = np.array(points)
        ptr_shape = points_arr.shape
        points_arr = points_arr.reshape(-1)

        """ convert end letter to the edge index of image """
        for i, point in enumerate(points_arr):
            if point == "end" and i % 2:
                """if end is on y-axis"""
                points_arr[i] = h
            elif point == "end" and not i % 2:
                points_arr[i] = w
        points_arr = points_arr.reshape(ptr_shape).astype(np.uint32)

        ptrs_black = []
        """ Find another end point of triangle """
        for point in points_arr:
            ptr1, ptr2 = point
            ptr3 = np.array([w, h]) * (ptr1 * ptr2 > 0)
            ptrs_black.append(np.vstack((ptr1, ptr2, ptr3)))

        cv2.fillPoly(self.img, ptrs_black, 0)

        return self

    def measureBright(self):
        """Measure the overall brighness of image."""
        blur = cv2.blur(self.img, (7, 7))
        imgHLS = cv2.cvtColor(blur, cv2.COLOR_BGR2HLS)
        self.bright = int(np.mean(imgHLS[:, :, 1]))
        return self

    def update_count(self, count: int):
        """Update the number of plants in the image."""
        self.count = count
        return self

    @property
    def shape(self) -> Tuple:
        """Return the shape of image."""
        return self.img.shape

    def astype(self, dtype) -> np.ndarray:
        """Change the type of image."""
        return self.img.astype(dtype)

    def __getitem__(self, item):
        """Return item."""
        return self.img.__getitem__(item)

    def exportPallete(self):
        """Return pallete."""
        return self.pallete

    def undistort(self, outpath=None):
        """Undistort image."""
        if self.param is None:
            logger.warning(
                "Distortion is not processe. Use the original image.")
            return self

        mtx = np.array(self.param["intrinsic"])
        dist = np.array(self.param["distortion coef."])
        h, w, _ = self.shape

        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist[:, :-1].squeeze(), (w, h), np.eye(3), balance=1
        )
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K=mtx,
            D=dist[:, :-1].squeeze(),
            R=np.eye(3),
            P=newcameramtx,
            size=(w, h),
            m1type=cv2.CV_32FC1,
        )
        dst = cv2.remap(
            self.img,
            map1,
            map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

        save_name = os.path.join(outpath, "undistort_" + self.name)
        cv2.imwrite(save_name, dst)
        logger.info(f"{save_name} is successfully saved.")

    def segment(
        self,
        config_file: str,
        checkpoint_file: str,
        show: bool = False,
        pallete_path: str = None,
        device="cuda:0",
    ):
        """Segmentation function

        Args:
            config_file (str): path of configuration file
            checkpoint_file (str): path of weight file
            show (bool, optional): whether to visualize. Defaults to False.
            pallete_path (str, optional): path of visualization. Defaults to None.
            device (str, optional): Defaults to "cuda:0".

        Returns:
            object itself
        """

        from mmseg.apis import inference_segmentor, init_segmentor

        model = init_segmentor(config_file, checkpoint_file, device=device)
        self.mask = inference_segmentor(model, self.img)

        if show:
            self.pallete = model.show_result(
                self.img,
                self.mask,
                out_file=os.path.join(pallete_path, "palatte_" + self.name),
                opacity=0.5,
            )

        return self

    def cropInstance(self, OUT_DIR: str) -> None:
        "Crop each instances and save them."
        if len(self.bbox[0]) == 0:
            return None

        for idx, bbox in enumerate(self.bbox[0]):
            x1, y1, x2, y2, _ = bbox
            cropImg = self.img[int(y1): int(y2), int(x1): int(x2)]
            cv2.imwrite(
                os.path.join(
                    OUT_DIR,
                    f"{os.path.splitext(self.name)[0]}_{idx}{os.path.splitext(self.name)[1]}",
                ),
                cropImg,
            )
        logger.info(
            f"{idx+1} crop image(s) is(are) generated from {self.name}.")

    def segment_with_model(self, show=False, pallete_path=None, filter=False):
        """
        Image segmentation based on MMsegmentation model is already mounted in self.

        TODO: write more
        """

        if self.cv_mode == "mmseg":

            from mmseg.apis import inference_segmentor

            self.mask = inference_segmentor(self.model, self.img)

            if show:
                """Save the segmentation image file"""
                self.model.show_result(
                    self.img,
                    self.mask,
                    out_file=os.path.join(
                        pallete_path, "palatte_" + self.name),
                    opacity=0.5,
                )
            else:
                self.pallete = self.model.show_result(
                    self.img, self.mask, opacity=0.5)

            return self

        elif self.cv_mode == "mmdet":
            from mmdet.apis import inference_detector

            bbox_result, segm_result = inference_detector(self.model, self.img)

            if filter:
                # if filter is true, use show only relevant prediction.

                bbox_result, segm_result = filter_prob(
                    bbox_result, segm_result, 0.5)
                bbox_result, segm_result = filter_center(
                    bbox_result, segm_result, 0.2)

            self.mask = segm_result[0]  # only care one class
            self.bbox = bbox_result

            if show:
                """Save the segmentation image file"""
                self.model.show_result(
                    self.img,
                    (bbox_result, segm_result),
                    out_file=os.path.join(
                        pallete_path, "palatte_" + self.name),
                )

            else:
                self.pallete = self.model.show_result(
                    self.img, (bbox_result, segm_result)
                )

            return self

        elif self.cv_mode == "mmcls":
            from mmcls.apis import inference_model

            self.health = {self.name: inference_model(self.model, self.img)}
            return self

    def mount(
        self,
        config_file,
        checkpoint_file,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        mode="mmseg",
    ):
        """Mount config file and weight file into the object."""

        if mode == "mmseg":
            from mmseg.apis import init_segmentor

            self.model = init_segmentor(
                config_file, checkpoint_file, device=device)

        elif mode == "mmdet":
            from mmdet.apis import init_detector

            self.model = init_detector(
                config_file, checkpoint_file, device=device)

        elif mode == "mmcls":
            from mmcls.apis import init_model

            self.model = init_model(
                config_file, checkpoint_file, device=device)

        else:
            ValueError("Unknown mode.")

        self.cv_mode = mode

        return self

    def compute_area(self) -> float:
        """Compute the actual area from mask image."""
        kernel = np.ones((21, 21), np.uint8)

        if self.cv_mode == "mmseg":
            if isinstance(self.mask, list):
                mask = self.mask[0]
            else:
                mask = self.mask

            output = cv2.morphologyEx(mask.astype(
                "uint8"), cv2.MORPH_OPEN, kernel)

            contours, _ = cv2.findContours(
                output, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )

            pixel_area = 0

            count = 0  # the number of plants
            thres = int(
                self.shape[0] * self.shape[1] / 10**3
            )  # if you want to change the sensitivity, then modify the value.

            for contour in contours:
                c_area = cv2.contourArea(contour)
                if c_area < thres:
                    """neglect too small mask"""
                    continue
                pixel_area += c_area
                count += 1

            # LAI is available only for semantic segmentation.
            self.maskRatio = pixel_area / self.img.shape[0] / self.img.shape[1]

        elif self.cv_mode == "mmdet":
            areas = []

            for seg in self.mask:
                areas.append(seg.sum())

            if len(areas) > 0:
                pixel_area = selectRepresentative(areas)

            else:
                "No relevant instance."
                pixel_area = 0

        """Area of leaf area in mm^2"""
        self.area = round(pixel_area * self.ratio)

        logger.info(f"Computed foreground area is: {self.area} mm2")
        return self

    def document(self, output: Dict):
        """Document output of process."""

        file_name, ext = os.path.splitext(os.path.basename(self.name))

        output[file_name] = {
            "ext": ext[1:],  # exclude dot.
            "area": self.area,
            "mode": self.cv_mode,
            "computed_at": strftime("%Y-%m-%d %H:%M:%S", gmtime()),
            "brightness": self.bright,
            "lai": -2*(math.log(self.maskRatio)*0.5*(math.pi/2))}

        # LAI_eff = 2*integral(0 to pi/2) -ln(maskRatio)*cos(theta)*sin(theta) d(theta)
        # Assuming


def selectRepresentative(areas: List) -> int:
    """select one value can represent area value

    Args:
        areas (List): List of areas

    Returns:
        int: one representative value
    """
    countAreas = len(areas)
    if countAreas < 5:
        return max(areas)

    elif 5 <= countAreas and 10 >= countAreas:
        # second biggest because the biggest could be wrong.
        return sorted(areas)[-2]
    else:
        return statistics.median(areas)


def filter_prob(bbox_result: List[List], segm_result: List[List], p: float):
    """Filter result with lower than certain confidence (p)."""

    idx2keep = []

    for idx, bbox in enumerate(bbox_result[0]):
        # only matter the first class in our case.
        if bbox[-1] > p:
            idx2keep.append(idx)

    bbox_result[0] = bbox_result[0][idx2keep]
    segm_result[0] = [seg for i, seg in enumerate(
        segm_result[0]) if i in idx2keep]

    return bbox_result, segm_result


def filter_center(bbox_result: List[List], segm_result: List[List], edge: float):
    """Filter the edge side of instances"""

    idx2keep = []

    if not any(segm_result):
        # if no segmentation mask is available.
        return bbox_result, segm_result

    else:
        h, w = segm_result[0][0].shape

    for idx, bbox in enumerate(bbox_result[0]):
        x1, y1, x2, y2, _ = bbox
        if (
            (x1 + x2) / 2 > edge * w
            and (x1 + x2) / 2 < (1 - edge) * w
            and (y1 + y2) / 2 > edge * h
            and (y1 + y2) / 2 < (1 - edge) * h
        ):
            # if center of instance is inside the boundary,
            idx2keep.append(idx)

    bbox_result[0] = bbox_result[0][idx2keep]
    segm_result[0] = [seg for i, seg in enumerate(
        segm_result[0]) if i in idx2keep]

    return bbox_result, segm_result


def _computeDisparity(
    stereo: object, img1_rect: np.ndarray, img2_rect: np.ndarray
) -> np.ndarray:
    return stereo.compute(img1_rect, img2_rect).astype(np.float32) / 16


def _check_color(file_name: Path):
    color_code = file_name.stem.split("-")[-2]
    assert color_code in [
        "rgb",
        "ir",
    ], f"Color code is neither rgb nor ir. It is {color_code}."
    return color_code


@dataclass
class HexaStereo:
    """Class for keeping track of stereo images."""

    distance: int  # distance between two cameras in mm
    cam_codes: str  # unique camera code
    base_depth: int  # manually measured depth to the pot in mm
    rgb: List[np.ndarray] = None  # rgb channel
    ir: List[np.ndarray] = None  # gray channel

    def registerRGB(self, filepaths: List[Path]) -> "HexaStereo":
        """
        Input: [rgb image 1, rgb image 2]
        """
        assert (
            len(filepaths) == 2
        ), f"Not two files are in the list. Here are files {filepaths}"
        assert self._check_color(filepaths[0]) == _check_color(
            filepaths[1]
        ), f"Error! Two different color codes are found."

        self.rgb = [
            cv2.cvtColor(cv2.imread(
                filepaths[0].__str__()), cv2.COLOR_BGR2RGB),
            cv2.cvtColor(cv2.imread(
                filepaths[1].__str__()), cv2.COLOR_BGR2RGB),
        ]
        return self

    def registerIR(self, filepaths: List[str]) -> "HexaStereo":
        """
        Input: [ir image 1, ir image 2]
        """
        assert (
            len(filepaths) == 2
        ), f"Not two files are in the list. Here are files {filepaths}"
        assert self._check_color(filepaths[0]) == _check_color(
            filepaths[1]
        ), f"Error! Two different color codes are found."

        self.ir = [
            cv2.cvtColor(cv2.imread(
                filepaths[0].__str__()), cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(cv2.imread(
                filepaths[1].__str__()), cv2.COLOR_BGR2GRAY),
        ]

        return self


@dataclass
class HexaPallelDepth(HexaStereo):
    """
    Class for keeping track of an depth eatimation.
    Assuming that two parallel(pre-rectified) images provided.
    """

    rgb_disparity_map: np.ndarray = None
    ir_disparity_map: np.ndarray = None

    def computeDisparity(
        self,
        show: bool = True,
        minDisparity: int = 10,
        numDisparities: int = 35,
        blockSize: int = 11,
    ):
        stereo = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
        )

        self.rgb_disparity_map = _computeDisparity(
            stereo, self.rgb[0], self.rgb[1])
        self.ir_disparity_map = _computeDisparity(
            stereo, self.ir[0], self.ir[1])

        if show:
            fig, ax = plt.subplots(figsize=(16, 8), nrows=2, ncols=2)
            ax[0, 0].set_title("Left image")
            ax[0, 0].imshow(self.rgb[0])
            ax[0, 1].set_title("Right image")
            ax[0, 1].imshow(self.rgb[1])

            ax[1, 0].set_title("Disparity map (RGB)")
            im3 = ax[1, 0].imshow(self.rgb_disparity_map)
            fig.colorbar(im3, ax=ax[1, 0], fraction=0.03, pad=0.04)

            ax[1, 1].set_title("Disparity map (IR)")
            im4 = ax[1, 1].imshow(self.ir_disparity_map)
            fig.colorbar(im4, ax=ax[1, 1], fraction=0.03, pad=0.04)
            plt.show()


@dataclass
class HexaNonPallelDepth(HexaStereo):
    """
    Class for keeping track of an depth eatimation.
    Assuming that two non-parallel images provided.
    Additional rig geometry information and rectification is needed compared to HexaPallelDepth.
    """

    pass
