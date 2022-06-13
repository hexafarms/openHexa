from dataclasses import dataclass
from typing import List, Dict, Union, Tuple, Optional
import numpy as np
import cv2
import glob
from loguru import logger
import os
import json
from json.decoder import JSONDecodeError
from numpyencoder import NumpyEncoder
import torch
from typing import Any


@dataclass
class hexa_img:
    """ image format handling Computer Vision task """
    img: np.ndarray = None
    mask: np.ndarray = None
    pallete: np.ndarray = None
    name: str= None
    param: Optional[Dict[str, int]] = None# camera parameter
    ratio: Optional[float] = 0 # cm2 per pixel
    area: Optional[float] = 0
    volume: Optional[float] = 0
    count: int = 1 # the number of plants in the bench
    model: Any = None

    # @property
    def load_img(self, filepath: str, metapath: str, separator):
        self.img = cv2.imread(str(filepath))
        assert type(self.img) != type(None), f"no file {filepath} exist!"

        self.name = os.path.basename(filepath)

        # load meta data
        with open(metapath, "r+") as j:
            try:
                data = json.load(j)
            except JSONDecodeError:
                logger.warning(
                    "Json file should not be empty. Delete the empty file and run again.")
                return 0
        # check if the value is inside meta.
        camera_code = separator.join(
            os.path.basename(filepath).split(separator)[:-1])

        if camera_code not in data.keys():
            logger.warning(f"no camera info in meta data.")
            return self

        if 'parameters' in data[camera_code].keys():
            logger.success(f"parameters of {filepath} is loaded.")
            self.param = data[camera_code]['parameters']
        else:
            logger.warning(f"parameters of {filepath} don't exist in {metapath}. no undistortion will be applied.")

        if 'pixel2cm' in data[camera_code].keys():
            logger.success(f"ratio of pixel to cm2 of {filepath} is loaded.")
            self.ratio = data[camera_code]['pixel2cm']
        else:
            logger.warning(
                f"ratio of pixel to cm2 of {filepath} don't exist in {metapath}. area will be in pixel level.")
            self.ratio = 1
        return self

    def remove(self,points:List):
        h, w, _ = self.shape
        points_arr = np.array(points)
        ptr_shape = points_arr.shape
        points_arr = points_arr.reshape(-1)

        """ convert end letter to the edge index of image """
        for i, point in enumerate(points_arr):
            if point == 'end' and i%2:
                """ if end is on y-axis """
                points_arr[i] = h
            elif point == 'end' and not i%2:
                points_arr[i] = w
        points_arr = points_arr.reshape(ptr_shape).astype(np.uint32)

        ptrs_black = []
        """ Find another end point of triangle """
        for point in points_arr:
            ptr1, ptr2 = point
            ptr3 = np.array([w, h]) * (ptr1 * ptr2 > 0)
            ptrs_black.append(np.vstack((ptr1,ptr2,ptr3)))
        
        cv2.fillPoly(self.img, ptrs_black, 0)

        return self

    def update_count(self, count:int):
        self.count = count
        return self

    @property
    def shape(self) -> Tuple:
        return self.img.shape

    def astype(self, dtype) -> np.ndarray:
        return self.img.astype(dtype)

    def __getitem__(self, item):
        return self.img.__getitem__(item)

    def undistort(self, outpath=None):
        """ undistort image """

        if self.param == None:
            logger.warning("distortion is not processed. Use the original image.")
            return self

        mtx = np.array(self.param['intrinsic'])
        dist = np.array(self.param['distortion coef.'])
        h, w, _ = self.shape

        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist[:, :-1].squeeze(), (w, h), np.eye(3), balance=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K=mtx, D=dist[:, :-1].squeeze(), R=np.eye(3), P=newcameramtx, size=(w, h), m1type=cv2.CV_32FC1)
        dst = cv2.remap(
            self.img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        if outpath == None:
            if distort_quality_check(dst):
                logger.success("undistortion is well processed.")
                self.img = dst
                return self
            else:
                logger.warning("undistortion failed. check the meta data.")
                logger.info("no undistortion due to inappropriate meta data")
                return self

        save_name = os.path.join(outpath, "undistort_"+self.name)
        cv2.imwrite(save_name, dst)
        logger.info(
            f"{save_name} is successfully saved.")

    def segment(self, config_file, checkpoint_file, show=False, pallete_path= None, device='cuda:0'):
        """
        image segmentation based on MMsegmentation
        config_file: configure file of MMsegmentation
        checkpoint_file: weight files

        TODO: write more
        """
        from mmseg.apis import inference_segmentor, init_segmentor
        model = init_segmentor(config_file, checkpoint_file, device=device)
        self.mask = inference_segmentor(model, self.img)

        if show:
            model.show_result(self.img, self.mask, out_file= os.path.join(pallete_path,"palatte_"+self.name), opacity=0.5)

        return self

    def segment_with_model(self, show=False, pallete_path= None):
        """
        image segmentation based on MMsegmentation
        model is already mounted in self.

        TODO: write more
        """
        """ This is for airflow api """
        import sys
        sys.path.append('/mmsegmentation')
        
        from mmseg.apis import inference_segmentor
        self.mask = inference_segmentor(self.model, self.img)

        if show:
            """ Save the segmentation image file """
            self.model.show_result(self.img, self.mask, out_file= os.path.join(pallete_path,"palatte_"+self.name), opacity=0.5)
        else:
            self.pallete = self.model.show_result(self.img, self.mask, opacity=0.5)


        return self

    def mount(self, config_file, checkpoint_file, device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        from mmseg.apis import init_segmentor
        self.model = init_segmentor(config_file, checkpoint_file, device=device)
        return self


    def compute_area(self) -> float:
        """ Compute the actual area from mask image """

        kernel = np.ones((21, 21), np.uint8)

        if type(self.mask)==type(list()):
            mask = self.mask[0]
        else:
            mask = self.mask

        output = cv2.morphologyEx(mask.astype(
            'uint8'), cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(output,
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        pixel_area = 0
        volume = 0
        count = 0 # the number of plants
        thres = int(self.shape[0]*self.shape[1]/10**3) # if you want to change the sensitivity, then modify the value.

        for contour in contours:
            c_area = cv2.contourArea(contour)
            if c_area < thres:
                """ neglect too small mask """
                continue
            pixel_area += c_area
            count += 1

            # # volume model 1 (assumption: half-sphere, separate plants)
            # mean_r = (c_area * self.ratio/np.pi)**0.5
            # volume += 2/3*np.pi*mean_r**3 
        # volume model 2 (assumtion: we know the number of separate plants)
        if count > self.count:
            """ If new plants are transplanted, then update the count value. """
            logger.info(f"Update the number of plants from {self.count} to {count}")
            self.count = count

        assert self.count != 0, "There is no plants in the image."

        if not self.ratio:
            logger.warning("No ratio between pixel to dimension. Output unit is in pixel.")
            self.ratio = 1
        volume = 2/3 * np.pi *((pixel_area /self.count* self.ratio/np.pi)**1.5) * self.count

        ''' area of leaf area in cm^2 '''
        self.area = round(pixel_area * self.ratio)
        self.volume = round(volume)
        logger.info(
            f"Computed foreground area is: {self.area} cm2, volume is {self.volume} cm3")
        return self
    
    def document(self, areas, graph=False, volume=True):
        if volume:
            areas.append([self.name, self.area, self.volume])
        else:
            areas.append([self.name, self.area])

        if graph:
            import matplotlib.pyplot as plt
            import pandas as pd
            import seaborn as sns
            import datetime

            df = pd.DataFrame(areas)
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])
            df['second'] = df['file_name'].apply(lambda x: x.split('-')[-1].split('.')[0])
            df['second'] = df['second'].astype('int64')
            df['hour'] = df['second'].subtract(df['second'].min()).div(3600)

            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(40,15))
            date = datetime.datetime.fromtimestamp(df['second'].iloc[-1]).strftime('%Y-%m-%d %H:%M:%S')
            fig.suptitle(f"Date of monitoring: {date}", fontsize=32)
            ax[0].set_title("Segmented Plant Image", fontsize=24)
            ax[0].axis('off')
            ax[0].imshow(self.pallete)
            ax[1].set_title("Plant Growth",fontsize=24)
            ax[1] = plt.gca()

            sns.lineplot(x = "hour", y= "area_cm2", data=df, linewidth = 3,  color='r',estimator=np.mean, legend="auto")
            ax[1].set_xlabel("hour", fontsize=16)
            ax[1].set_ylabel("area (unit: cm2)", fontsize=16)
            plt.savefig(os.path.join("output",f'{date}.png'))




class hexa_process:
    """ image processing to get meta data """

    def __init__(self):
        self.camera_code = None

    def calibrate(self, imgpath_checker, corner_w, corner_h, metafile, separator="-"):
        objpoints = []
        imgpoints = []
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        winSize = (11, 11)
        zeroZone = (-1, -1)
        objp = np.zeros((corner_h*corner_w, 3), np.float32)
        objp[:, :2] = np.mgrid[0:corner_h,
                               0:corner_w].T.reshape(-1, 2)
        images = glob.glob(imgpath_checker + '/*')

        for fname in images:
            if self.camera_code == None:
                self.camera_code = separator.join(os.path.basename(
                    fname).split(separator)[:-1])

            elif self.camera_code != separator.join(os.path.basename(fname).split(separator)[:-1]):
                logger.warning(
                    f"Every checker image should have same camera code: camera code{separator}image number. If there is multiple separator, the last separator is counted.")
                break
            else:
                pass

            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            self.size = gray.shape[::-1]

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(
                gray, (corner_h, corner_w), None)

            # If found, add object points, image points (after refining them)
            if ret == True:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(
                    gray, corners, winSize, zeroZone, criteria)  # Increase the accuracy
                imgpoints.append(corners2)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(
                    img, (corner_h, corner_w), corners2, ret)
                cv2.namedWindow("img", cv2.WINDOW_NORMAL)
                cv2.imshow('img', img)
                cv2.waitKey()
            cv2.destroyAllWindows()

        fixpts = 3
        ret, mtx, dist, _, _, _ = cv2.calibrateCameraRO(
            objpoints, imgpoints, gray.shape[::-1], fixpts, None, None)

        if ret < 0.5:
            logger.info(
                f"Your RMS re-projection error is {ret}. This is acceptable.")
            value = {"intrinsic": mtx,
                     "distortion coef.": dist}
            self._update_meta(self.camera_code+separator, value, metafile,
                              separator=separator, mode="parameters")

        else:
            logger.info(
                f"Your RMS re-projection error is {ret}. Inacceptable!. Use the better quality of checker board images.")

        return self

    def compute_px_ratio(self, filepath: str, metapath: str, separator, actual_dim,  debug=True) -> float:
        default_par1 = 200
        default_par2 = 30
        logger.info(f"{filepath} will be processed.")

        #TODO: if calibration information is available in meta, use the undistort image. if not, use the original image. 

        with open(os.path.join(metapath, "hexa_meta.json"), "r+") as j:
            try:
                data = json.load(j)
            except JSONDecodeError:
                logger.warning(
                    "Json file should not be empty. Delete the empty file and run again.")
                return 0

        src = cv2.imread(filepath, cv2.IMREAD_COLOR)
        if self.camera_code == None:
            self.camera_code = separator.join(os.path.basename(
                    filepath).split(separator)[:-1])

        if self.camera_code in data.keys() and "parameters" in data[self.camera_code].keys():
            mtx = np.array(data[self.camera_code]["parameters"]["intrinsic"])
            dist = np.array(data[self.camera_code]["parameters"]["distortion coef."])
            undist_src = self._undistort(src, mtx, dist)
            if distort_quality_check(undist_src):
                src = undist_src

        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        radius = self._extract_circle(gray, par1=default_par1,
                                      par2=default_par2, debug=debug)

        if radius == 0:
            logger.info(
                f"{filepath} is not applicable for circle Hough transform.")
            return None

        ratio = self._compute_ratio(radius, actual_dim)
        self._update_meta(filepath, ratio, metapath,
                          separator, mode="pixel2cm",)

    @staticmethod
    def _undistort(img, mtx, dist):
        """ undistort image """

        h, w, _ = img.shape

        newcameramtx = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            mtx, dist[:, :-1].squeeze(), (w, h), np.eye(3), balance=1)
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            K=mtx, D=dist[:, :-1].squeeze(), R=np.eye(3), P=newcameramtx, size=(w, h), m1type=cv2.CV_32FC1)
        dst = cv2.remap(
            img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        return dst


    def _extract_circle(self, src, par1: float, par2: float, debug=True, count=0) -> float:
        """
        par1: it is the higher threshold of the two passed to the Canny edge detector (the lower one is twice smaller)
        par2: it is the accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected. 
        """

        height, width = src.shape[:2]

        logger.info(
            f"Extract circles using parameters: par1: {par1}, par2: {par2}")

        circles = cv2.HoughCircles(src, cv2.HOUGH_GRADIENT, 1, height / 24,
                                   param1=par1, param2=par2,
                                   minRadius=width//48, maxRadius=width//24)

        if circles is not None and debug:
            temp_src = np.dstack((src, src, src))
            circles = np.uint16(np.around(circles))
            color = np.random.choice(range(256), size=3)
            for i in circles[0, :]:
                center = (i[0], i[1])
                # circle center
                cv2.circle(temp_src, center, 3, (0, 100, 100), 3)
                # circle outline
                radius = i[2]
                cv2.circle(temp_src, center, radius,
                           (int(color[0]), int(color[1]), int(color[2])), 3)
            cv2.imshow("detected circles", temp_src)
            cv2.waitKey(0)

        if count > 30:
            logger.info("Can't find representative circles.")
            return 0

        if circles is None:
            logger.info("No circles are detected. Parameters will be loose.")

            return self._extract_circle(src, par1=par1*np.random.uniform(low=0.6, high=0.8), par2=par2*np.random.uniform(low=0.6, high=0.8), count=count+1)

        num_circle = len(circles.squeeze())

        if num_circle < 4:
            logger.info(
                f"Not enough number of circles are detected. ({num_circle} circles) Parameters will be loose.")
            return self._extract_circle(src, par1=par1*np.random.uniform(low=0.7, high=1.1), par2=par2*np.random.uniform(low=0.7, high=1.1), count=count+1)

        # if it doesn't converge, increase the value to higher than 2.
        elif np.std(circles.squeeze()[:, 2]) > 2:
            logger.info(
                "Too many circles with different size. Parameters will be strict.")
            return self._extract_circle(src, par1=par1*np.random.uniform(low=0.9, high=1.3), par2=par2*np.random.uniform(low=0.9, high=1.3), count=count+1)

        else:
            circle_radius = np.median(circles, axis=1)[0, 2]
            logger.info(f"Found the right parameters!: {circle_radius}")
            return circle_radius

    @staticmethod
    def _compute_ratio(radius: float, ACTUAL_DIM: float) -> float:
        """
        radius: radius in pixel unit
        ACTUAL_DIM: actual dimension of diameter in cm unit

        return: pixel 2 cm^2
        """
        diameter = radius * 2
        ratio = (ACTUAL_DIM/diameter)**2
        logger.info("Ratio of cm2/px is computed: "+str(ratio))

        return round(ratio, 4)

    @staticmethod
    def _update_meta(filepath: str, value: float or Dict[str, np.ndarray], meta_loc: str, separator: str, mode: str, ):

        assert mode == "pixel2cm" or "parameters", "Wrong variable. It should be either pixel2cm or parameters."

        if meta_loc:
            """ If no meta data, create a new one. """
            if not os.path.isdir(meta_loc):
                os.mkdir(meta_loc)

            loc_meta = os.path.join(meta_loc, "hexa_meta.json")
            key_img_name = separator.join(
                os.path.basename(filepath).split(separator)[:-1])

            meta = {key_img_name:
                    {
                        mode: value
                    }
                    }

            if os.path.exists(loc_meta):
                """ Append to the existing one"""
                with open(loc_meta, "r+") as j:
                    try:
                        data = json.load(j)
                    except JSONDecodeError:
                        logger.warning(
                            "Json file should not be empty. Delete the empty file and run again.")
                        return 0

                if key_img_name in data.keys():
                    """ No information of the device. """
                    if mode in data[key_img_name]:
                        logger.info("Already exist meta data of this image")

                        if type(value) in [int, float, np.float64]:
                            # if pix2dim,
                            diff = abs(data[key_img_name]
                                       [mode] - value) > value * 0.2
                        else:
                            # if camera parameter
                            diff = np.sum(np.sum(a) for a in (np.array(list(value.items()), dtype=object)[
                                          :, 1] - np.array(list(data[key_img_name][mode].items()), dtype=object)[:, 1])) > 10

                        if diff:
                            logger.warning(
                                f"value of this image is different to old value. Old: {data[key_img_name][mode]}, New: {value}. Keep the old one.")
                            if int(input("Want to update? update: 1, no update: 0 \n")):
                                logger.info(
                                    f"Update value of {key_img_name} to {value}")
                                data[key_img_name][mode] = value
                                with open(loc_meta, "w") as k:
                                    json.dump(data, k, indent=4,
                                              cls=NumpyEncoder)

                        else:
                            logger.info(
                                f"No update the value of {key_img_name}.")
                    else:
                        data[key_img_name][mode] = value
                        with open(loc_meta, "w") as k:
                            json.dump(data, k, indent=4, cls=NumpyEncoder)
                        logger.info(f"{mode} of {key_img_name} is saved.")

                else:
                    logger.info(f"Update value of {key_img_name} to {value}")
                    data.update(meta)
                    with open(loc_meta, "w") as k:
                        json.dump(data, k, indent=4, cls=NumpyEncoder)

            else:
                logger.info(f"Generate a new meta data.")
                with open(loc_meta, 'w') as f:
                    json.dump(meta, f, indent=4, cls=NumpyEncoder)

def distort_quality_check(img: np.ndarray)->bool:
    """ Check if the undistorted image is correct given the input is from fish-eye camera. """

    h, w, _ = img.shape
    h_edge = h//40
    w_edge = w//40
    
    if img[:h_edge, w_edge].sum() + img[-h_edge:, -w_edge:].sum() + img[:h_edge, -w_edge:].sum() + img[-h_edge:, :w_edge].sum() == 0:
        """ Undistortion failed """
        logger.warning("Distortion has failed. Check the calibration process again.")
        return False

    else:
        logger.success("Distortion is well done.")
        return True