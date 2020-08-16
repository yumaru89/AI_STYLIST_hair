import os

import numpy as np

def has_img_ext(fname):
    ext = os.path.splitext(fname)[1]
    return ext in ('.jpg', '.jpeg', '.png')


def make_same_size(img1, img2):
    im1_h, im1_w, im1_z = img1.shape[:3]
    im2_h, im2_w, im2_z = img2.shape[:3]
    h_diff = np.abs(im1_h - im2_h)
    w_diff = np.abs(im1_w - im2_w)
    large_h = im1_h if im1_h > im2_h else im2_h

    if im1_h > im2_h:
        h_im2_fill = np.zeros((h_diff, im2_w, im2_z), dtype=np.uint8)
        img2 = np.append(img2, h_im2_fill, axis=0)
    else:
        h_im1_fill = np.zeros((h_diff, im1_w, im1_z), dtype=np.uint8)
        img1 = np.append(img1, h_im1_fill, axis=0)

    if im1_w > im2_w:
        w_im2_fill = np.zeros((large_h, w_diff, im2_z), dtype=np.uint8)
        img2 = np.append(img2, w_im2_fill, axis=1)
    else:
        w_im1_fill = np.zeros((large_h, w_diff, im1_z), dtype=np.uint8)
        img1 = np.append(img1, w_im1_fill, axis=1)

    return img1, img2
