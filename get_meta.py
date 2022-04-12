from hexa_img import hexa_process

if __name__ == "__main__":

    INPUT = "images/dev-0-1640275201.jpg"
    PATHCHECKER = "checker"
    SEPARATOR = "-"
    METAPATH = "."
    CORNER_WIDTH = 9
    CORNER_HEIGHT = 6
    ACTUAL_DIM = 5  # CM

    hexa = hexa_process()
    hexa.calibrate(imgpath_checker=PATHCHECKER, corner_w=CORNER_WIDTH,
                   corner_h=CORNER_HEIGHT, metafile=METAPATH, separator=SEPARATOR)
    
    hexa.compute_px_ratio(filepath=INPUT,
                          metapath=METAPATH, separator=SEPARATOR, actual_dim=ACTUAL_DIM)
    
