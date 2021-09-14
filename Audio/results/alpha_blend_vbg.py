import os
import re
import sys
import cv2
import numpy as np
from glob import glob

regex = re.compile(r'\d+\.png')


def fill_hole(trans):
    im_th = trans[..., 0]
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
    
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
    cv2.floodFill(im_floodfill, mask, (0,0), 255)
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    im_out = im_th | im_floodfill_inv
    return im_out[:, :, None]
 

def alpha_blend_vbg(srcdirbg, srcdir):
    files = glob(os.path.join(srcdir, "*.png"))
    files = [x for x in files if regex.match(os.path.basename(x)) is not None]
    files = sorted(files, key=lambda x: int(os.path.basename(os.path.splitext(x)[0])))

    for i, fpath in enumerate(files):
        tarfpath = fpath[:-4] + "_blend2.png"
        # if os.path.exists(tarfpath):
        #     continue
        im1 = cv2.imread(os.path.join(srcdirbg, os.path.basename(fpath)))
        im2 = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        trans = fill_hole(im2[:, :, -1:]).astype(np.float32) / 255.0
        im2 = im2[:, :, :-1]
        im3 = im1.astype(np.float32) * (1-trans) + im2.astype(np.float32) * trans
        im3 = im3.astype(np.uint8)
        cv2.imwrite(tarfpath, im3)
        cv2.imshow('blend', im3)
        cv2.waitKey(1)


alpha_blend_vbg(sys.argv[1], sys.argv[2])
