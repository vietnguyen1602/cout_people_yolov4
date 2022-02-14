# -*- coding: utf-8 -*-
"""
Class definition of YOLO_v4 style detection model on image and video
"""

import colorsys
import time

import tensorflow as tf
from tensorflow.compat.v1.keras import backend as K
from keras.backend.tensorflow_backend import get_session
import numpy as np
from keras import backend as K
from keras.models import load_model
import cv2
from yolo4.model import yolo_eval, Mish
from yolo4.utils import letterbox_image
import os
from keras.utils import multi_gpu_model

class YOLO(object):
    def __init__(self):
        self.model_path = 'model_data/yolo4.h5'
        self.anchors_path = 'model_data/yolo_anchors.txt'
        self.classes_path = 'model_data/coco_classes.txt'
        self.gpu_num = 1
        while True:
            t = time.localtime()
            if (t.tm_hour > 20) or (t.tm_hour < 8):
                self.score = 0.6
                break
            else:
                self.score = 0.35
                break
        self.iou = 0.4
        self.class_names = self._get_class()
        self.anchors = self._get_anchors()
        self.sess = tf.compat.v1.keras.backend.get_session()
        self.model_image_size = (416, 416)  # fixed size or (None, None)
        self.is_fixed_size = self.model_image_size != (None, None)
        self.boxes, self.scores, self.classes = self.generate()

    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def _get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            anchors = np.array(anchors).reshape(-1, 2)
        return anchors

    def generate(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.yolo_model = load_model(model_path, custom_objects={'Mish': Mish}, compile=False)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))
        np.random.seed(10101)  # Fixed seed for consistent colors across runs.
        np.random.shuffle(self.colors)  # Shuffle colors to decorrelate adjacent classes.
        np.random.seed(None)  # Reset seed to default.

        # Generate output tensor targets for filtered bounding boxes.
        self.input_image_shape = K.placeholder(shape=(2, ))
        if self.gpu_num>=2:
            self.yolo_model = multi_gpu_model(self.yolo_model, gpus=self.gpu_num)
        boxes, scores, classes = yolo_eval(self.yolo_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score, iou_threshold=self.iou)
        return boxes, scores, classes

    def detect_image(self, image):
        person = 0
        if self.is_fixed_size:
            assert self.model_image_size[0]%32 == 0, 'Multiples of 32 required'
            assert self.model_image_size[1]%32 == 0, 'Multiples of 32 required'
            boxed_image = letterbox_image(image, tuple(reversed(self.model_image_size)))
        else:
            new_image_size = (image.width - (image.width % 32),
                              image.height - (image.height % 32))
            boxed_image = letterbox_image(image, new_image_size)
        image_data = np.array(boxed_image, dtype='float32')

        # print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })
        return_boxes = []
        return_scores = []
        return_class_names = []
        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            if predicted_class != 'person':  # Modify to detect other classes.
                continue
            else:
                person +=1
            box = out_boxes[i]
            score = out_scores[i]
            x = int(box[1])
            y = int(box[0])
            w = int(box[3] - box[1])
            h = int(box[2] - box[0])
            if x < 0:
                w = w + x
                x = 0
            if y < 0:
                h = h + y
                y = 0
            return_boxes.append([x, y, w, h])
            return_scores.append(score)
            return_class_names.append(predicted_class)
        return return_boxes, return_scores, return_class_names, person

    # def light_feedback(self, image):
    #     img_resize = cv2.resize(image, (256, 256))
    #     img_resize = cv2.cvtColor(img_resize, cv2.COLOR_BGR2HSV)
    #     avg_color_per_row = np.average(img_resize, axis=0)
    #     avg_color = np.average(avg_color_per_row, axis=0)
    #     val = avg_color[2]
    #     if (val >= 100):
    #         return True
    #     else:
    #         return False

    def close_session(self):
        self.sess.close()
