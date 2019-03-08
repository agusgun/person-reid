from .tabs.detection_tracking_tab import DetectionTrackingTab
from .tabs.input_tab import InputTab
from .tabs.reidentification_tab import ReidentificationTab
from PyQt5.QtWidgets import QTabWidget, QVBoxLayout, QWidget

class TabWidget(QWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.input_tab = InputTab()
        self.detection_tracking_tab = DetectionTrackingTab(self.input_tab.input_widget.th_input)
        self.reid_tab = ReidentificationTab()
        self.tabs.addTab(self.input_tab, "Input")
        self.tabs.addTab(self.detection_tracking_tab, "Detection and Tracking")
        self.tabs.addTab(self.reid_tab, "Re-identification")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)

