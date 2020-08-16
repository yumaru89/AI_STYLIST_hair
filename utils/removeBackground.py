import os
from io import BytesIO

import numpy as np
from PIL import Image
import cv2

import tensorflow as tf
import datetime


class DeepLabModel(object):
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'

    def __init__(self, tarball_path):
        self.graph = tf.Graph()

        graph_def = None
        graph_def = tf.compat.v1.GraphDef.FromString(
            open(tarball_path + "/frozen_inference_graph.pb", "rb").read())

        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')

        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')

        self.sess = tf.compat.v1.Session(graph=self.graph)

    def run(self, image):

        start = datetime.datetime.now()
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        end = datetime.datetime.now()
        diff = end - start
        print("Time taken to evaluate segmentation is : " + str(diff))

        resized_image_rgba = image.convert('RGBA').resize(target_size, Image.ANTIALIAS)
        return resized_image_rgba, seg_map


def drawSegment(baseImg, matImg):
    width, height = baseImg.size
    dummyImg = np.zeros([height, width, 4], dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = matImg[y, x]
            (r, g, b, a) = baseImg.getpixel((x, y))
            if color == 0 :
                dummyImg[y, x, 3] = 0
            else:
                dummyImg[y, x] = [r, g, b, a]
    return dummyImg


def run_visualization(user_img):
    try:
        user_img = user_img.copy()
        user_img = cv2.cvtColor(user_img, cv2.COLOR_BGRA2RGBA)
        user_img = Image.fromarray(user_img)
        # orignal_im = Image.open(filepath).convert("RGBA")
    except IOError:
        print('Cannot retrieve image. Please check file: ')
        return

    modelType = "models/xception_model"
    MODEL = DeepLabModel(modelType)
    resized_im, seg_map = MODEL.run(user_img)

    back_removal_img = drawSegment(resized_im, seg_map)

    return back_removal_img
