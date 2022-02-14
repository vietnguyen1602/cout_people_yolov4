import cv2
import numpy as np
import imutils
import os
import  time
# mt_1 = np.zeros((3, 3))
# mt_2 = np.zeros((3, 3))
# mt_1[:][:] = 1
# mt_2[:][:] = 2
# mt_3 = np.zeros((3, 3))
#
# # print(mt_1)
# # print(mt_2)
#
# for i in range(3):
#     for j in range(3):
#         a = mt_1[i][j]
#         b = mt_2[i][j]
#         mt_temp = [a, b]
#         c = max(mt_temp)
#         mt_3[i][j] = c
# print(mt_3)



# file_path = 'rtsp://admin:admin123@10.10.105.15:554/cam/realmonitor?channel=1&subtype=0'
# cap = cv2.VideoCapture(file_path)
# frame_count = 0
# while True:
#     _, frame = cap.read()
#     #frame = cv2.resize(frame, (960, 640))
#     # frame = cv2.rotate(frame, cv2.cv2.ROTATE_180)
#     frame_count += 1
#     if(frame_count == 1):
#         cv2.imshow('', frame)
#         cv2.waitKey(0)
#         cv2.imwrite('camera_id_11.jpg', frame)
#         print('done')
#         break

# img = cv2.imread('image/grid_map_plot.png')
# img = cv2.resize(img, (640, 480))
# cv2.imwrite('grid_map.jpg', img)


# while 1:
#     t = time.localtime()
#     if ((t.tm_sec == 30) or (t.tm_sec == 59)):
#         print(1)
#         time.sleep(2)
#


cap = cv2.VideoCapture('rtsp://admin:admin123@10.10.105.15:554/cam/realmonitor?channel=1&subtype=0')
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
frame_size = (frame_width, frame_height)
fps = 20

output = cv2.VideoWriter('Resources/output_video_from_file.avi', cv2.VideoWriter_fourcc('M','J','P','G'), 20, frame_size)