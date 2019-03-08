from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea, QHBoxLayout, QVBoxLayout, QWidget
import os

class ReidentificationWidget(QScrollArea):
    def __init__(self, parent=None):
        super(ReidentificationWidget, self).__init__(parent)
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignTop)
        
        dir_path = os.path.dirname(__file__)
        filename1 = os.path.join(dir_path, '../frame_output/1_0.png')
        filename2 = os.path.join(dir_path, '../frame_output/1_20.png')

        pixmap1 = QPixmap(filename1)
        pixmap2 = QPixmap(filename2)

        label1 = QLabel()
        label2 = QLabel()

        label1.setPixmap(pixmap1)
        label2.setPixmap(pixmap2)

        image_layout = QHBoxLayout()
        image_layout.addWidget(label1)
        image_layout.addWidget(label2)

        for index in range(100):
            layout.addWidget(QLabel('Person %02d' % index))
            if index == 0:
                layout.addLayout(image_layout)
        
        self.setWidget(widget)
        self.setWidgetResizable(True)

