from app.thread.detection_reidentification_thread import DetectionReidentificationThread
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget


class DetectionReidentificationTab(QWidget):
    def __init__(self, input_thread, parent=None):
        super(DetectionReidentificationTab, self).__init__(parent)
        self.input_thread = input_thread

        self.label = QLabel()
        self.label.resize(800, 600)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.th_detection_reidentification = DetectionReidentificationThread(input_thread)
        self.th_detection_reidentification.changePixmap.connect(self.setImage)
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

