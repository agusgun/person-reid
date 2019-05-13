from app.thread.detection_reidentification_thread import DetectionReidentificationThread
from app.widget.reidentification_panel_widget import ReidentificationPanelWidget
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QHBoxLayout, QVBoxLayout, QWidget


class DetectionReidentificationTab(QWidget):
    def __init__(self, input_thread, parent=None):
        super(DetectionReidentificationTab, self).__init__(parent)
        self.input_thread = input_thread

        self.label = QLabel()
        self.label.resize(800, 600)

        reidentification_panel_widget = ReidentificationPanelWidget()
        
        self.layout = QVBoxLayout()
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.addWidget(self.label)
        self.layout.addWidget(reidentification_panel_widget)
        self.setLayout(self.layout)

        self.th_detection_reidentification = DetectionReidentificationThread(input_thread)
        self.th_detection_reidentification.changePixmap.connect(self.setImage)
        self.th_detection_reidentification.update_display_trigger.connect(reidentification_panel_widget.add_new_person)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

