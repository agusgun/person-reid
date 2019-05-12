from .tabs.detection_tracking_tab import DetectionTrackingTab
from .tabs.detection_reidentification_tab import DetectionReidentificationTab
from .tabs.input_tab import InputTab
from .tabs.reidentification_tab import ReidentificationTab
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

import os
class TabWidget(QWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.input_tab = InputTab()
        
        self.tabs.addTab(self.input_tab, "Input")
        if bool(os.environ['PERSON_REID_DIRECT_REIDENTIFICATION']):
            self.detection_reidentification_tab = DetectionReidentificationTab(self.input_tab.input_widget.th_input)
            self.tabs.addTab(self.detection_reidentification_tab, "Detection and Re-identification")
        else:
            self.detection_tracking_tab = DetectionTrackingTab(self.input_tab.input_widget.th_input)
            self.reid_tab = ReidentificationTab()
            self.tabs.addTab(self.detection_tracking_tab, "Detection and Tracking")
            self.tabs.addTab(self.reid_tab, "Re-identification")
            self.detection_tracking_tab.th_detection_tracking.change_person_id.connect(
                self.reid_tab.th_reidentification.set_person_id)

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)