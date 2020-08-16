import numpy as np
import cv2
import dlib

CASCADE = cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
PREDICTOR = dlib.shape_predictor('./models/shape_predictor_81_face_landmarks.dat')

def facemark(gray_img):
    faces_roi = CASCADE.detectMultiScale(gray_img, minSize=(100, 100))

    face_lms = []
    for _ in faces_roi:
        detector = dlib.get_frontal_face_detector()
        rects = detector(gray_img, 1)
        face_lms = []
        for rect in rects:
            face_lms.append(
                np.array([[p.x, p.y] for p in PREDICTOR(gray_img, rect).parts()]))
    return face_lms
