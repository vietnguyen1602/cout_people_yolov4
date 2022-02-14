from __future__ import division, print_function, absolute_import
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO
import modul
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync
import socketio
import os
import time

warnings.filterwarnings('ignore')
# sio = socketio.Client()
# sio.connect('http://10.10.102.225:1000', namespaces='/camera')
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# config = tf.config.experimental.set_memory_growth(physical_devices[0], True)
def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    # Definition of the camera
    path_camera_11 = 0
    path_camera_12 = 1
    path_camera_13 = 2
    path_camera_14 = 3
    path_camera_15 = 4
    # load video
    cap_11 = cv2.VideoCapture(path_camera_11)
    cap_12 = cv2.VideoCapture(path_camera_12)
    cap_13 = cv2.VideoCapture(path_camera_13)
    cap_14 = cv2.VideoCapture(path_camera_14)
    cap_15 = cv2.VideoCapture(path_camera_15)
    while True:
        # đọc dữ liệu từ các camera
        _, camera_11 = cap_11.read()
        _, camera_12 = cap_12.read()
        _, camera_13 = cap_13.read()
        _, camera_14 = cap_14.read()
        _, camera_15 = cap_15.read()
        # resize
        t = time.localtime()
        frame_count += 1
        if ((t.tm_sec)%30 == 0):                        #(frame_count == 1) or ((frame_count%450) == 0):
            camera_11 = cv2.resize(camera_11, (960, 640))
            camera_12 = cv2.resize(camera_12, (960, 640))
            camera_13 = cv2.resize(camera_13, (960, 640))
            camera_14 = cv2.resize(camera_14, (960, 640))
            camera_15 = cv2.resize(camera_15, (960, 640))
            # # draw grid
            modul.draw_grid_total(camera_11, os.path.join('npy', 'horizontal_id_11.npy'),
                                  os.path.join('npy', 'vertical_id_11.npy'))
            modul.draw_grid_total(camera_12, os.path.join('npy', 'horizontal_id_12.npy'),
                                  os.path.join('npy', 'vertical_id_12.npy'))
            modul.draw_grid_total(camera_13, os.path.join('npy', 'horizontal_id_13.npy'),
                                  os.path.join('npy', 'vertical_id_13.npy'))
            modul.draw_grid_total(camera_14, os.path.join('npy', 'horizontal_id_14.npy'),
                                  os.path.join('npy', 'vertical_id_14.npy'))
            modul.draw_grid_total(camera_15, os.path.join('npy', 'horizontal_id_15.npy'),
                                  os.path.join('npy', 'vertical_id_15.npy'))
            # pre-processing
            image_id_11 = Image.fromarray(camera_11[..., ::-1])  # bgr to rgb
            image_id_12 = Image.fromarray(camera_12[..., ::-1])  # bgr to rgb
            image_id_13 = Image.fromarray(camera_13[..., ::-1])  # bgr to rgb
            image_id_14 = Image.fromarray(camera_14[..., ::-1])  # bgr to rgb
            image_id_15 = Image.fromarray(camera_15[..., ::-1])  # bgr to rgb
            # detection
            boxes_11, confidence_11, classes_11, person_11 = yolo.detect_image(image_id_11)
            boxes_12, confidence_12, classes_12, person_12 = yolo.detect_image(image_id_12)
            boxes_13, confidence_13, classes_13, person_13 = yolo.detect_image(image_id_13)
            boxes_14, confidence_14, classes_14, person_14 = yolo.detect_image(image_id_14)
            boxes_15, confidence_15, classes_15, person_15 = yolo.detect_image(image_id_15)
            # lấy tọa độ người và ảnh đầu ra
            camera_11, person_id_11 = modul.detection(boxes_11, confidence_11, classes_11, person_11, camera_11)
            camera_12, person_id_12 = modul.detection(boxes_12, confidence_12, classes_12, person_12, camera_12)
            camera_13, person_id_13 = modul.detection(boxes_13, confidence_13, classes_13, person_13, camera_13)
            camera_14, person_id_14 = modul.detection(boxes_14, confidence_14, classes_14, person_14, camera_14)
            camera_15, person_id_15 = modul.detection(boxes_15, confidence_15, classes_15, person_15, camera_15)
            # lấy ma trận số người của từng camera
            matrix_id_11 = modul.local_matrix_id_11(person_id_11)
            matrix_id_12 = modul.local_matrix_id_12(person_id_12)
            matrix_id_13 = modul.local_matrix_id_13(person_id_13)
            matrix_id_14 = modul.local_matrix_id_14(person_id_14)
            matrix_id_15 = modul.local_matrix_id_15(person_id_15)
            map_matrix = matrix_id_11 + matrix_id_12 + matrix_id_13 + matrix_id_14 + matrix_id_15
            #sio.emit('Camera', map_matrix, namespace='/camera')
            print(map_matrix)
            # vẽ bản đồ
            grid_map = modul.heatmap2d(map_matrix)
            # show image
            camera_11 = cv2.resize(camera_11, (480, 320))
            camera_12 = cv2.resize(camera_12, (480, 320))
            camera_13 = cv2.resize(camera_13, (480, 320))
            camera_14 = cv2.resize(camera_14, (480, 320))
            camera_15 = cv2.resize(camera_15, (480, 320))
            cv2.imshow('11', camera_11)
            cv2.imshow('12', camera_12)
            cv2.imshow('13', camera_13)
            cv2.imshow('14', camera_14)
            cv2.imshow('15', camera_15)
            cv2.imshow('head_map', grid_map)
            cv2.waitKey(1)
            print('gui loi')
            # giải phóng bộ nhớ


    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(YOLO())
