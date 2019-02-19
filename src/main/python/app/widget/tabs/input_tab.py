from app.widget.input_widget import InputWidget
from PyQt5.QtWidgets import QVBoxLayout, QWidget

class InputTab(QWidget):
    def __init__(self, parent=None):
        super(InputTab, self).__init__(parent)

        self.input_widget = InputWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.input_widget)
        self.setLayout(layout)