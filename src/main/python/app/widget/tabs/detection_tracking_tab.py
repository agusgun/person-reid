from app.thread.detection_tracking_thread import DetectionTrackingThread
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class DetectionTrackingTab(QWidget):
    def __init__(self, input_thread, parent=None):
        super(DetectionTrackingTab, self).__init__(parent)
        self.input_thread = input_thread

        self.label = QLabel()
        self.label.resize(800, 600)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.th_detection_tracking = DetectionTrackingThread(input_thread)
        self.th_detection_tracking.changePixmap.connect(self.setImage)
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))
