import cv2
import numpy as np
from PIL import Image
import imutils.video

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.offsetbox import TextArea, DrawingArea, OffsetImage, AnnotationBbox
import time

# arr_coor = []
# img = cv2.imread('camera_id_11.jpg')
# def get_coor(event,x,y,flags,param):
#     if (event == cv2.EVENT_LBUTTONDBLCLK):
#         cv2.putText(img,f'{x ,y}', (x, y-5), 1, 2, (0, 0, 255), 2)
#         cv2.circle(img, (x, y), 10, (255, 255, 255), -1)
#         coor = [x,y]
#         arr_coor.append(coor)
#         print(arr_coor)
#         # np.array(arr_coor).dump(open('grid_map.npy', 'wb'))
# cv2.namedWindow('image')
# cv2.setMouseCallback('image', get_coor)
# while(1):
#     cv2.imshow('image', img)
#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# map_matrix = np.load(open('npy/grid_map.npy', 'rb'), allow_pickle=True)
# map_matrix[5]  = [386, 369]
# map_matrix[6] = [386, 369]
# map_matrix[7] = [386, 272] #272
# map_matrix[8] = [386, 272]
# map_matrix[9] = [131, 272]
# map_matrix[10] = [131, 272]
# map_matrix[11] = [131, 203]
# map_matrix[12] = [131, 203]
# map_matrix[13] = [81, 203]
# map_matrix[14] = [81, 203]
# map_matrix[15] = [81, 115]
# # print(map_matrix)
# np.array(map_matrix).dump(open('grid_map.npy', 'wb'))

# import time
# t = time.localtime()
# t.tm_sec
# print(t.tm_sec)
# sio = socketio.Client()
# sio.connect('http://10.10.102.225:1000', namespaces='/camera')
# map_matrix = np.load(open('npy/map.npy', 'rb'), allow_pickle=True)
# # flatten_matrix = map_matrix.flatten()
# map_matrix = map_matrix.astype(int)
# human_list = map_matrix.tolist()
# # print(type(flatten_matrix))
# # x = [[1, 2, 3], [4, 4, 5]]
# # print(type(x))
# # while True:
# sio.emit('Camera', human_list, namespace='/camera')
# with open('grid_map.jpg', 'rb') as f:
#     image_data = f.read()
# print(image_data)
#
# img_data_2 = cv2.imread('grid_map.jpg') as f
# print(img_data_2)
# emit('my-image-event', {'image_data': image_data})
# while True:
#     t = time.localtime()
#     if (t.tm_hour < 20):
#         print(t.tm_hour)
#         break
