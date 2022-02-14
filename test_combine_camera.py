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

# sio = socketio.Client()
# sio.connect('http://10.10.104.4:1000', namespaces='/camera')
warnings.filterwarnings('ignore')
# map = np.zeros((640, 960, 3), dtype=np.uint8)
person_coor = []
global_map = np.zeros((4, 12, 3), np.uint8)
def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    tracking = False
    file_path = 'rtsp://admin:admin123@10.10.105.11:554/cam/realmonitor?channel=1&subtype=0'
    file_path_2 = 'rtsp://admin:admin123@10.10.105.13:554/cam/realmonitor?channel=1&subtype=0'
    cap1 = cv2.VideoCapture(file_path)
    cap2 = cv2.VideoCapture(file_path_2)
    ###
    while True:
        # đọc ảnh
        _, frame = cap1.read()
        _, frame_2 = cap2.read()
        frame = cv2.resize(frame, (960, 640))
        frame_2 = cv2.resize(frame_2, (960, 640))
        frame_count += 1
        if (frame_count == 1 or (frame_count % 100) == 0):
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            image_2 = Image.fromarray(frame_2[..., ::-1])  # bgr to rgb
            modul.draw_grid_2(frame_2)
            modul.draw_grid(frame)
            boxes, confidence, classes, person = yolo.detect_image(image)
            boxes_2, confidence_2, classes_2, person_2 = yolo.detect_image(image_2)
            frame, person_coor = modul.detection(boxes, confidence, classes, person, frame)
            frame_2, person_coor_2 = modul.detection(boxes_2, confidence_2, classes_2, person_2, frame_2)
            map_matrix, temp_map = modul.local_map(person_coor, global_map)
            map_matrix, temp_map = modul.local_map_2(person_coor_2, map_matrix, global_map)
            global_map_resize = cv2.resize(temp_map, (1200, 400))
            # hiển thị
            frame = cv2.resize(frame, (480, 320))
            frame_2 = cv2.resize(frame_2, (480, 320))
            cv2.imshow('', frame)
            cv2.imshow('farm_2', frame_2)
            cv2.imshow('global_map', global_map_resize)
            print(map_matrix)

            # Press Q to stop!
            cv2.waitKey(1)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main(YOLO())
