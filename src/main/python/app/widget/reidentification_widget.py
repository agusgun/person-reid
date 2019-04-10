from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QLabel, QScrollArea, QHBoxLayout, QVBoxLayout, QWidget
import os
import datetime

class ReidentificationWidget(QScrollArea):
    def __init__(self, parent=None):
        super(ReidentificationWidget, self).__init__(parent)
        widget = QWidget()
        self.layout = QVBoxLayout(widget)
        self.layout.setAlignment(Qt.AlignTop)
        
        self.input_dir_path = os.path.join(os.path.dirname(__file__), '../keyframe_output')
        
        self.setWidget(widget)
        self.setWidgetResizable(True)
    
    @pyqtSlot(int, int)
    def add_new_person(self, person_id, keyframe_id):
        person_layout = QVBoxLayout()
        person_layout.addWidget(QLabel('Person %02d' % person_id))
        
        face_keyframe_dir_path = os.path.join(self.input_dir_path, str(keyframe_id))
        face_keyframe_paths = os.listdir(face_keyframe_dir_path)
        face_keyframe_paths = [os.path.join(face_keyframe_dir_path, path) for path in face_keyframe_paths]

        face_keyframe_layout = QHBoxLayout()
        for img_path in face_keyframe_paths:
            pixmap = QPixmap(img_path)
            label_image = QLabel()
            label_image.setPixmap(pixmap)
            face_keyframe_layout.addWidget(label_image)
        person_layout.addLayout(face_keyframe_layout)

        curr_datetime = str(datetime.datetime.now())
        person_layout.addWidget(QLabel('Keyframe ID %02d on %s' % (keyframe_id, curr_datetime)))

        self.layout.addLayout(person_layout)

    @pyqtSlot(int, int)
    def update_datetime_person(self, person_id, keyframe_id):
        print('update')
        curr_datetime = str(datetime.datetime.now())
        new_label = QLabel('Keyframe ID %02d on %s' % (keyframe_id, curr_datetime))
        self.layout.itemAt(person_id - 1).addWidget(new_label)