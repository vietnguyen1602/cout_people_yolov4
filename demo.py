
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
# sio.connect('http://10.10.102.225:1000', namespaces='/camera')
warnings.filterwarnings('ignore')

def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
    frame_count = 0
    #Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    tracking = False
    writeVideo_flag = True
    asyncVideo_flag = False
    file_path = 'rtsp://admin:admin123@10.10.105.11:554/cam/realmonitor?channel=1&subtype=0'
    # file_path_2 = 'rtsp://admin:admin123@10.10.105.13:554/cam/realmonitor?channel=1&subtype=0'
    video_capture = cv2.VideoCapture(file_path)
    while True:
        map = np.zeros((640, 960, 3), dtype=np.uint8)
        person_coor = []
        ret, frame = video_capture.read()  # frame shape 640*480*3
        frame_count +=1
        if(frame_count == 1 or (frame_count % 100) == 0):
            frame = cv2.resize(frame, (960, 640))
            image = Image.fromarray(frame[..., ::-1])  # bgr to rgb
            modul.draw_grid(frame)
            modul.draw_grid(map)
            boxes, confidence, classes, person = yolo.detect_image(image)
            status = modul.light_feedback(frame)
            data = {"number_of_people": person, "status_of_light": status}
            # sio.emit('Camera', data, namespace='/camera')
            if tracking:
                features = encoder(frame, boxes)
                detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                              zip(boxes, confidence, classes, features)]
            else:
                detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                              zip(boxes, confidence, classes)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]
            if tracking:
                # Call the tracker
                tracker.predict()
                tracker.update(detections)

                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255), 2)
                    cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1.5e-3 * frame.shape[0], (0, 255, 0), 1)
            for det in detections:
                bbox = det.to_tlbr()
                score = "%.2f" % round(det.confidence * 100, 2) + "%"
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                x, y = modul.draw_point(bbox)
                coor = [x, y]
                person_coor.append(coor)
                cv2.circle(map, (int(x), int(y)), 8, (0, 0, 255), -1)
                if len(classes) > 0:
                    cls = det.cls
                    cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 2)), (int(bbox[0]) + 90,
                                            int(bbox[1] - 20)), (0, 0, 0), -1)
                    # str(cls) + " " +
                    cv2.putText(frame, score, (int(bbox[0]), int(bbox[1] - 2)), 0,
                                0.8, (255, 255, 255), 2)
                    cv2.rectangle(frame, (5, 50), (5 + 300, 50 - 25), (0, 0, 0), -1)
                    cv2.putText(frame, f'Total Persons : {person}', (5, 50), 1, 2, (255, 255, 255), 2)
            # map_matrix, temp_map = modul.local_map(person_coor)
            # global_map = cv2.resize(temp_map, (1200, 400))
            cv2.imshow('', frame)
            # cv2.imshow('map', map)
            # cv2.imshow('global', global_map)
            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(1)
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main(YOLO())
