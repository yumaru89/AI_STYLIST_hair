import os
import sys
import pickle

import numpy as np
import cv2
import torch

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.facemark import facemark
from utils.utils import has_img_ext
from utils.prepare_mask import prepare_hair_mask, prepare_network


def main():
    model_shape_pickle = './model_data/model_shape.pickle'
    os.makedirs("./model_data/", exist_ok=True)

    img_dir = "./img/model/resize_face"

    img_paths = [os.path.join(img_dir, k) for k in sorted(os.listdir(img_dir)) if has_img_ext(k)]
    img_names = [os.path.splitext(os.path.basename(img_path))[0] for img_path in img_paths]

    facelm_lists = faceShape(img_paths)
    model_shape_dic = dict(zip(img_names, facelm_lists))

    with open(model_shape_pickle, "wb") as f:
        pickle.dump(model_shape_dic, f)

    hairShape(img_paths)

    cv2.destroyAllWindows()


def hairShape(img_paths):
    save_dir = './img/model/hair'
    os.makedirs(save_dir, exist_ok=True)

    test_image_transforms, net = prepare_network()

    with torch.no_grad():
        for i, img_path in enumerate(img_paths, 1):
            hair_mask = prepare_hair_mask(img_path, test_image_transforms, net)

            face_image = cv2.imread(img_path)
            bgr = cv2.split(face_image)
            bgra = cv2.merge(bgr + [hair_mask])
            save_path = os.path.join(save_dir, os.path.splitext(os.path.basename(img_path))[0]+'.png')
            cv2.imwrite(save_path, bgra)


def faceShape(img_paths):
    face_lm_list = []

    for img_path in img_paths:
        user_face = cv2.imread(img_path)
        user_face_gray = cv2.cvtColor(user_face, cv2.COLOR_BGR2GRAY)
        face_lms = facemark(user_face_gray)

        if len(face_lms) > 1:
            print("顔複数", img_path)
            face_lm_list.append([])
            continue
        if not len(face_lms):
            print("顔なし", img_path)
            face_lm_list.append([])
            continue

        face_lm_list.append(face_lms[0].tolist())

        for i, points in enumerate(face_lms[0]):
            cv2.drawMarker(user_face, (points[0], points[1]), (21, 255, 12), markerSize=5)

        save_path = os.path.join('test/model_mark', os.path.splitext(os.path.basename(img_path))[0] + '.png')

        cv2.imwrite(save_path, user_face)

    return face_lm_list


if __name__ == '__main__':
    main()

