from __future__ import division, print_function, absolute_import
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
import socketio
import matplotlib.pyplot as plt

def detection(boxes, confidence, classes, person, frame):
    ###
    nms_max_overlap = 1.0
    person_coor = []
    ###
    detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in zip(boxes, confidence, classes)]
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
    detections = [detections[i] for i in indices]
    for det in detections:
        bbox = det.to_tlbr()
        score = "%.2f" % round(det.confidence * 100, 2) + "%"
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
        x, y = draw_point(bbox)
        coor = [x, y]
        person_coor.append(coor)
        np.array(person_coor).dump(open('person_id_13.npy', 'wb'))
        cv2.circle(frame, (int(x), int(y)), 8, (0, 0, 255), -1)
        # cv2.circle(map, (int(x), int(y)), 8, (0, 0, 255), -1)
        # cv2.putText(map, f'{int(x), int(y)}', (int(x), int(y) - 2), 0,
        #             0.8, (255, 255, 255), 2)
        if len(classes) > 0:
            cls = det.cls
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1] - 2)), (int(bbox[0]) + 90,
                                                                    int(bbox[1] - 20)), (0, 0, 0), -1)
            cv2.putText(frame, score, (int(bbox[0]), int(bbox[1] - 2)), 0,
                        0.8, (255, 255, 255), 2)
            cv2.rectangle(frame, (5, 50), (5 + 300, 50 - 25), (0, 0, 0), -1)
            cv2.putText(frame, f'Total Persons : {person}', (5, 50), 1, 2, (255, 255, 255), 2)
    return frame, person_coor

def light_feedback(image):
    img_resize = cv2.resize(image, (256, 256))
    img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    avg_color_per_row = np.average(img_resize, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    val = avg_color[2]
    if (val >= 100):
        return True
    else:
        return False
def draw_grid_total(img, array_path_horizonltal, array_path_vertical):
    myArray_horizontal = np.load(open(array_path_horizonltal,'rb'), allow_pickle=True)
    myArray_vertical = np.load(open(array_path_vertical, 'rb'), allow_pickle=True)
    for i in range(len(myArray_horizontal)):
        if ((i % 2) == 1):
            start_point = myArray_horizontal[i - 1]
            end_point = myArray_horizontal[i]
            cv2.line(img, (start_point[0], start_point[1]),
                     (end_point[0], end_point[1]), (255, 255, 255), 2)
    for j in range(len(myArray_vertical)):
        if ((j % 2) == 1):
            start_point = myArray_vertical[j - 1]
            end_point = myArray_vertical[j]
            cv2.line(img, (start_point[0], start_point[1]),
                     (end_point[0], end_point[1]), (255, 255, 255), 2)

def draw_point(bbox):
    start_point = [bbox[0], bbox[1]]
    end_point = [bbox[2], bbox[3]]
    x = (start_point[0]+end_point[0])/2
    y = end_point[1]
    return x, y

def check_point(A, B, C):
    x1 = A[0]
    y1 = A[1]
    x2 = B[0]
    y2 = B[1]
    x = C[0]
    y = C[1]
    check = (y1-y2)*(x-x1)+(x2-x1)*(y-y1)
    if (check < 0):
        return 0
    else:
        return 1
def color(x):
    switcher = {
        0: (132, 132, 132),
        1: (0, 255, 0),
        2: (0, 255, 191),
        3: (0, 128, 255),
        4: (0, 0, 255),
        5: (46, 46, 254),
        6: (46, 46, 254),
        7: (46, 46, 254)
    }
    return switcher.get(x, "nothing")
def local_matrix_id_11(per_coor):
    horizontal_coor = np.load(open('npy/horizontal_id_11.npy', 'rb'), allow_pickle=True)
    vertical_coor = np.load(open('npy/vertical_id_11.npy', 'rb'), allow_pickle=True)
    map_matrix = np.zeros([7, 11])
    col = 0
    row = 0
    for j in range(len(per_coor)):
        C = [per_coor[j][0], per_coor[j][1]]
        hor_count = 0
        for i in range(len(horizontal_coor)):
            if ((i + 1) % 2 == 0):
                A = [horizontal_coor[i - 1][0], horizontal_coor[i - 1][1]]
                B = [horizontal_coor[i][0], horizontal_coor[i][1]]
                check_hor = check_point(A, B, C)
                if (check_hor == 1):
                    col = hor_count + 8
                    break
                hor_count += 1
        ver_count = 0
        for k in range(len(vertical_coor)):
            if ((k + 1) % 2 == 0):
                D = [vertical_coor[k - 1][0], vertical_coor[k - 1][1]]
                E = [vertical_coor[k][0], vertical_coor[k][1]]
                check_ver = check_point(D, E, C)
                if (check_ver == 1):
                    row = ver_count + 4
                    break
                ver_count += 1
        map_matrix[row][col] += 1
    return map_matrix
def local_matrix_id_12(per_coor):
    horizontal_coor = np.load(open('npy/horizontal_id_12.npy', 'rb'), allow_pickle=True)
    vertical_coor = np.load(open('npy/vertical_id_12.npy', 'rb'), allow_pickle=True)
    map_matrix = np.zeros([7, 11])
    col = 0
    row = 0
    for j in range(len(per_coor)):
        C = [per_coor[j][0], per_coor[j][1]]
        hor_count = 0
        # print('nguoi', j, '  ', C)
        for i in range(len(horizontal_coor)):
            if ((i + 1) % 2 == 0):
                A = [horizontal_coor[i - 1][0], horizontal_coor[i - 1][1]]
                B = [horizontal_coor[i][0], horizontal_coor[i][1]]
                check_hor = check_point(A, B, C)
                if (check_hor == 1):
                    row = hor_count + 2
                    break
                hor_count += 1
        ver_count = 0
        for k in range(len(vertical_coor)):
            if ((k + 1) % 2 == 0):
                D = [vertical_coor[k - 1][0], vertical_coor[k - 1][1]]
                E = [vertical_coor[k][0], vertical_coor[k][1]]
                check_ver = check_point(D, E, C)
                if (check_ver == 0):
                    col = ver_count + 2
                    break
                if ((check_ver == 1) and (ver_count == 2)):
                    col = 5
                    break
                ver_count += 1
        map_matrix[row][col] += 1
    return map_matrix
def local_matrix_id_13(per_coor):
    horizontal_coor = np.load(open('npy/horizontal_id_13.npy', 'rb'), allow_pickle=True)
    vertical_coor = np.load(open('npy/vertical_id_13.npy', 'rb'), allow_pickle=True)
    map_matrix = np.zeros([7, 11])
    col = 0
    row = 0
    for j in range(len(per_coor)):
        C = [per_coor[j][0], per_coor[j][1]]
        hor_count = 0
        for i in range(len(horizontal_coor)):
            if ((i + 1) % 2 == 0):
                A = [horizontal_coor[i - 1][0], horizontal_coor[i - 1][1]]
                B = [horizontal_coor[i][0], horizontal_coor[i][1]]
                check_hor = check_point(A, B, C)
                if (check_hor == 0):
                    col = hor_count + 7
                    break
                hor_count += 1
                if ((check_hor == 1) and (hor_count == 2)):
                    col = 9
                    break
        ver_count = 0
        for k in range(len(vertical_coor)):
            if ((k + 1) % 2 == 0):
                D = [vertical_coor[k - 1][0], vertical_coor[k - 1][1]]
                E = [vertical_coor[k][0], vertical_coor[k][1]]
                check_ver = check_point(D, E, C)
                if (check_ver == 1):
                    row = ver_count
                    break
                ver_count += 1
        if (row == 0) and (col ==0):
            break
        else:
            map_matrix[row][col] += 1
    return map_matrix
def local_matrix_id_14(per_coor):
    horizontal_coor = np.load(open('npy/horizontal_id_14.npy', 'rb'), allow_pickle=True)
    vertical_coor = np.load(open('npy/vertical_id_14.npy', 'rb'), allow_pickle=True)
    map_matrix = np.zeros([7, 11])
    col = 0
    row = 0
    for j in range(len(per_coor)):
        C = [per_coor[j][0], per_coor[j][1]]
        hor_count = 0
        # print('nguoi', j, '  ', C)
        for i in range(len(horizontal_coor)):
            if ((i + 1) % 2 == 0):
                A = [horizontal_coor[i - 1][0], horizontal_coor[i - 1][1]]
                B = [horizontal_coor[i][0], horizontal_coor[i][1]]
                check_hor = check_point(A, B, C)
                if (check_hor == 1):
                    col = 2
                else:
                    col = 3
                    ver_count = 0
        for k in range(len(vertical_coor)):
            if ((k + 1) % 2 == 0):
                D = [vertical_coor[k - 1][0], vertical_coor[k - 1][1]]
                E = [vertical_coor[k][0], vertical_coor[k][1]]
                check_ver = check_point(D, E, C)
                if (check_ver == 1):
                    row = 1
                else:
                    row = 0
        map_matrix[row][col] += 1
    return map_matrix
def local_matrix_id_15(per_coor):
    horizontal_coor = np.load(open('npy/horizontal_id_15.npy', 'rb'), allow_pickle=True)
    vertical_coor = np.load(open('npy/vertical_id_15.npy', 'rb'), allow_pickle=True)
    map_matrix = np.zeros([7, 11])
    col = 0
    row = 0
    for j in range(len(per_coor)):
        C = [per_coor[j][0], per_coor[j][1]]
        hor_count = 0
        # print('nguoi', j, '  ', C)
        for i in range(len(horizontal_coor)):
            if ((i + 1) % 2 == 0):
                A = [horizontal_coor[i - 1][0], horizontal_coor[i - 1][1]]
                B = [horizontal_coor[i][0], horizontal_coor[i][1]]
                check_hor = check_point(A, B, C)
                if (check_hor == 0):
                    col = hor_count + 2
                    break
                if (check_hor == 1) and (hor_count == 2):
                    col = 4
                    break
        ver_count = 0
        for k in range(len(vertical_coor)):
            if ((k + 1) % 2 == 0):
                D = [vertical_coor[k - 1][0], vertical_coor[k - 1][1]]
                E = [vertical_coor[k][0], vertical_coor[k][1]]
                check_ver = check_point(D, E, C)
                if (check_ver == 1):
                    row = 5
                    break
                else:
                    row = 4
                    break
        map_matrix[row][col] += 1
    return map_matrix
# hàm vẽ bản đồ
def global_map(map_matrix):
    global_map = np.zeros((7, 11, 3), np.uint8)
    # for a in range(7):
    #     for b in range(11):
    #         global_map[a][b] = (255, 255, 255)
    for i in range(len(map_matrix)):
        for j in range(len(map_matrix[0])):
            x = int(map_matrix[i][j])
            color_value = color(x)
            global_map[i][j] = color_value
    return global_map
def heatmap2d(arr: np.ndarray):
    fig = plt.figure()
    plt.imshow(arr, cmap='YlGnBu')
    plt.colorbar()
    fig.canvas.draw()
    img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8,
                        sep='')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    myArray_horizontal = np.load(open('npy/grid_map.npy', 'rb'), allow_pickle=True)
    for i in range(len(myArray_horizontal)):
        if ((i % 2) == 1):
            start_point = myArray_horizontal[i - 1]
            end_point = myArray_horizontal[i]
            cv2.line(img, (start_point[0], start_point[1]),
                     (end_point[0], end_point[1]), (0, 0, 0), 2)
    return img
