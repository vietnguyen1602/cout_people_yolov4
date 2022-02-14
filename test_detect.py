from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
import imutils.video
from videocaptureasync import VideoCaptureAsync

warnings.filterwarnings('ignore')

# Definition of the parameters
max_cosine_distance = 0.3
nn_budget = None
nms_max_overlap = 1.0
def main(yolo):
   # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    tracking = True
    writeVideo_flag = True
    asyncVideo_flag = False


image = cv2.imread('frame23.jpg')
image = Image.fromarray(image[...,::-1])  # bgr to rgb
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes, confidence, classes = YOLO.detect_image(image)
detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]
boxes = np.array([d.tlwh for d in detections])
scores = np.array([d.confidence for d in detections])
indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
detections = [detections[i] for i in indices]
for det in detections:
    bbox = det.to_tlbr()
    score = "%.2f" % round(det.confidence * 100, 2) + "%"
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
    if len(classes) > 0:
        cls = det.cls
        cv2.putText(image, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                    1.5e-3 * image.shape[0], (0, 255, 0), 1)
cv2.imshow('', image)
cv2.waitKey(0)
