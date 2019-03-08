from PyQt5.QtWidgets import QVBoxLayout, QWidget, QLabel
from app.widget.reidentification_widget import ReidentificationWidget

class ReidentificationTab(QWidget):
    def __init__(self, parent=None):
        super(ReidentificationTab, self).__init__(parent)

        label_test = QLabel("hehehe")
        reidentification_widget = ReidentificationWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        layout.addWidget(label_test)
        layout.addWidget(reidentification_widget)
        self.setLayout(layout)
        