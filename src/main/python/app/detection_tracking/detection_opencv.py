import cv2 as cv
import numpy as np
import os

class DetectionOpenCV:
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

        self.net = cv.dnn.readNetFromDarknet(self.modelConfiguration, self.modelWeights)
        self.net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
    
    # Get the names of the output layers
    def getOutputsNames(self, net):
        # Get the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Get the names of the output layers, i.e. the layers with unconnected outputs
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

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

        classIds = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > self.confThreshold:
                    center_x = int(detection[0] * frameWidth)
                    center_y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

        boxes_for_tracking = []
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            if classIds[i] == 0: #check if class equal to person
                box = boxes[i]
                left = box[0]
                top = box[1]
                width = box[2]
                height = box[3]
                if left < 0:
                    width += left
                    left = 0
                if top < 0:
                    height += top
                    top = 0
                boxes_for_tracking.append([left, top, width, height])
            
        return boxes_for_tracking

    