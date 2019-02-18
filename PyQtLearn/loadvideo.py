# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QTabWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QPixmap, QImage
import sys
import os
import cv2 as cv
import numpy as np

class InputThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, video=False):
        super().__init__()
        self.is_running = True
        self.ret = None
        self.cap = None

    def run(self):
        while True and self.is_running:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                rgbImage = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.msleep(40)
        print('Cap Rel')
        self.cap.release()

    def set_file_name(self, file_name):
        self.file_name = file_name
    
    def get_capture(self):
        if self.ret:
            return self.ret, self.frame
        else:
            return None, None

    def change_input2video(self):
        self.cap = cv.VideoCapture(self.file_name)

    def change_input2camera(self):
        self.cap = cv.VideoCapture(0)

    def signal_start(self):
        self.is_running = True

    def signal_stop(self):
        self.is_running = False

class InputWidget(QWidget):
    def __init__(self, parent=None):
        super(InputWidget, self).__init__(parent)

        self.label = QLabel()
        self.label.resize(800, 600)

        self.th_input = InputThread()
        self.th_input.changePixmap.connect(self.setImage)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def start_camera(self):
        self.th_input.signal_stop()
        self.th_input.change_input2camera()
        self.th_input.signal_start()
        self.th_input.start()

    def start_video(self):
        self.th_input.signal_stop()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video",
            QDir.homePath())
        if file_name != '':
            self.th_input.set_file_name(file_name)
            self.th_input.change_input2video()
            self.th_input.signal_start()
            self.th_input.start()

class InputTab(QWidget):
    def __init__(self, parent=None):
        super(InputTab, self).__init__(parent)

        self.input_widget = InputWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.input_widget)
        self.setLayout(layout)

class Detection:
    def __init__(self):
        self.confThreshold = 0.5
        self.nmsThreshold = 0.4
        self.inpWidth = 416
        self.inpHeight = 416
        self.classesFile = "../coco.names"
        self.classes = None
        with open(self.classesFile, 'rt') as f:
            self.classes = f.read().rstrip('\n').split('\n')
        self.modelConfiguration = "../yolov3.cfg"
        self.modelWeights = "../yolov3.weights"

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
    def drawPred(self, frame, classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)
        
        label = '%.2f' % conf
            
        # Get the label for the class name and its confidence
        if self.classes:
            assert(classId < len(self.classes))
            label = '%s:%s' % (self.classes[classId], label)

        #Display the label at the top of the bounding box
        labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv.rectangle(frame, (left, top - round(1.5*labelSize[1])), (left + round(1.5*labelSize[0]), top + baseLine), (255, 255, 255), cv.FILLED)
        cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,0), 1)

    # Remove the bounding boxes with low confidence using non-maxima suppression
    def postprocess(self, frame, outs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]

        classIds = []
        confidences = []
        boxes = []
        # Scan through all the bounding boxes output from the network and keep only the
        # ones with high confidence scores. Assign the box's class label as the class with the highest score.
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

        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(boxes, confidences, self.confThreshold, self.nmsThreshold)
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            self.drawPred(frame, classIds[i], confidences[i], left, top, left + width, top + height)


class DetectionTrackingThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        self.detection = Detection()

    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                # Detection Stuff Here
                # Create a 4D blob from a frame.
                blob = cv.dnn.blobFromImage(frame, 1/255, (self.detection.inpWidth, self.detection.inpHeight), [0,0,0], 1, crop=False)
                # Sets the input to the network
                self.detection.net.setInput(blob)
                # Runs the forward pass to get output of the output layers
                outs = self.detection.net.forward(self.detection.getOutputsNames(self.detection.net))
                # Remove the bounding boxes with low confidence
                self.detection.postprocess(frame, outs)
                # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = self.detection.net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                
                # Convert frame to display to PyQt
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                # self.msleep(1000)

    def signal_start(self):
        self.is_running = True
        self.start()

    def signal_stop(self):
        self.is_running = False

class DetectionTrackingTab(QWidget):
    def __init__(self, input_thread, parent=None):
        super(DetectionTrackingTab, self).__init__(parent)
        self.input_thread = input_thread

        self.label = QLabel()
        self.label.resize(800, 600)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(QPushButton('hehe'))
        self.setLayout(self.layout)        

        self.th_detection_tracking = DetectionTrackingThread(input_thread)
        self.th_detection_tracking.changePixmap.connect(self.setImage)
        # self.th_detection_tracking.start()

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


class TabWidget(QWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.input_tab = InputTab()
        self.detection_tracking_tab = DetectionTrackingTab(self.input_tab.input_widget.th_input)
        self.reid_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Input")
        self.tabs.addTab(self.detection_tracking_tab, "Detection and Tracking")
        self.tabs.addTab(self.reid_tab, "Re-identification")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)


class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Person Re-ID")

        # Menu Bar
        use_camera_action = QAction(QIcon('camera.png'), '&Use Camera', self)
        use_camera_action.setShortcut('Ctrl+E')
        use_camera_action.setStatusTip('Use camera')
        use_camera_action.triggered.connect(self.use_camera)

        open_action = QAction(QIcon('open.png'), '&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open video')
        open_action.triggered.connect(self.open_file)

        exit_action = QAction(QIcon('exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.exit_call)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(use_camera_action)
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        self.tabs = TabWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)

    def use_camera(self):
        self.tabs.input_tab.input_widget.start_camera()
        self.tabs.detection_tracking_tab.th_detection_tracking.signal_start()

    def open_file(self):
        self.tabs.input_tab.input_widget.start_video()
        self.tabs.detection_tracking_tab.th_detection_tracking.signal_start()

    def exit_call(self):
        sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())