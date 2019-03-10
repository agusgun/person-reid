from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel
from app.widget.reidentification_widget import ReidentificationWidget
from app.thread.reidentification_thread import ReidentificationThread

class ReidentificationTab(QWidget):
    def __init__(self, parent=None):
        super(ReidentificationTab, self).__init__(parent)
        
        reidentification_widget = ReidentificationWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(reidentification_widget)
        self.setLayout(layout)
        
        self.th_reidentification = ReidentificationThread()
        self.th_reidentification.start()