import cv2 as cv
import numpy as np
import os
from pydarknet import Detector

class Detection:
    def __init__(self):
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4
        self.inpWidth = 416
        self.inpHeight = 416
        package_path = os.path.abspath(os.path.dirname(__file__))
        self.classesFile = os.path.join(package_path, "../model_data/coco.data")
        self.classesName = os.path.join(package_path, "../model_data/coco.names")
        self.classes = None
        with open(self.classesName, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.modelConfiguration = os.path.join(package_path, "../model_data/yolov3.cfg")
        self.modelWeights = os.path.join(package_path, "../model_data/yolov3.weights")

        self.net = Detector(bytes(self.modelConfiguration, encoding='utf-8'), bytes(self.modelWeights, encoding='utf-8'), 0, 
                            bytes(self.classesFile, encoding='utf-8'))

    # Draw the predicted bounding box
    def drawPred(self, frame, class_name, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
        label = '%s:%s' % (class_name, label)
        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)
        # print(top, left, bottom, right)

    # Convert bounding box format for tracking purpose
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        boxes_for_tracking = []
        for class_name, score, bounds in outs:
            class_name = class_name.decode('utf-8')
            if class_name != 'person':
                continue
            center_x, center_y, width, height = bounds
            center_x = int(center_x)
            center_y = int(center_y)
            width = int(width)
            height = int(height)
            left = int(center_x - width / 2)
            top = int(center_y - height / 2)
            
            if left < 0:
                width += left
                left = 0
            if top < 0:
                height += top
                top = 0

            # self.drawPred(frame, class_name, score, left, top, left + width, top + height)
            boxes_for_tracking.append([left, top, width, height])
            # print('wei', width, height)

        return boxes_for_tracking