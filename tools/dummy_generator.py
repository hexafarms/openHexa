import cv2, os
import numpy as np
import matplotlib.pyplot as plt

def dummy_generator(W,H,typ):
    """ Generate a dummy image which mimics an undistorted image. """
    intensity = 128
    dummy = np.zeros(( H//2, W//2, 3 ), dtype=np.uint8)
    h, w, _ = dummy.shape # shape of the quater dummy
    
    if typ == "Barrel":
        dummy = cv2.circle(dummy, (w,h), int(max(h, w)), (intensity,intensity,intensity), -1)

    elif typ == "Pincushion":
        dummy = cv2.circle(dummy, (w,-h), int(h*1.2), (intensity,intensity,intensity), -1)
        dummy = cv2.circle(dummy, (-w,h), int(w*1.1), (intensity,intensity,intensity), -1)
    else:
        print('no appripriate distortion type! should select in [Barrel, Pincushion]')

    """ mirror images """
    full_dummy = np.zeros(( H, W, 3 ), dtype = np.uint8)
    full_dummy [ :h, :w, ...] = dummy
    full_dummy [ -h:, -w:, ...] = cv2.flip(dummy, -1)
    full_dummy [ :h, -w:, ...] = cv2.flip(dummy, 1)
    full_dummy [ -h:, :w, ...] = cv2.flip(dummy, 0)

    if typ == "Pincushion":
        full_dummy = cv2.bitwise_not(full_dummy) - (255-intensity)

    return full_dummy
        
if __name__ == "__main__":
    W = 1280
    H = 720
    typs = ["Barrel", "Pincushion"]

    for typ in typs:

        assert typ in ["Barrel", "Pincushion"]
        img = dummy_generator(W,H,typ)
        cv2.imwrite (os.path.join("demo", typ+".png"), img)