from app.deep_sort.detection import Detection as ddet
from app.detection_tracking.detection import Detection
import cv2 as cv
import os
from pydarknet import Image
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage
import time

class DetectionReidentificationThread(QThread):
    changePixmap = pyqtSignal(QImage)
    change_person_id = pyqtSignal(int)
    
    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        self.detection = Detection()
        self.person_iterator_dict = dict()
        self.counter = 0
        
    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                start_time = time.time()

                # Detection Stuff Here
                dark_frame = Image(frame)
                outs = self.detection.net.detect(dark_frame, thresh=self.detection.confThreshold, nms=self.detection.nmsThreshold)
                del dark_frame
                
                # Remove the bounding boxes with low confidence
                bboxes_for_detection = self.detection.postprocess(frame, outs)
                self.counter += 1
                for bbox in bboxes_for_detection:
                    left, top, width, height = bbox
                    cv.rectangle(frame, (int(left), int(top)), (int(left + width), int(top+height)), (255,255,255), 2)

                end_time = time.time()
                # Put efficiency information
                label = 'Inference time: %.2f ms' % ((end_time - start_time)*1000)
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                # Convert frame to display to PyQt
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                
    def signal_start(self):
        self.is_running = True
        self.start()

    def signal_stop(self):
        self.is_running = False
