from app.thread.input_thread import InputThread
from PyQt5.QtCore import QDir, pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QImage, QPixmap
class InputWidget(QWidget):
    def __init__(self, parent=None):
        super(InputWidget, self).__init__(parent)

        self.label = QLabel()
        self.label.resize(800, 600)

        self.th_input = InputThread()
        self.th_input.changePixmap.connect(self.setImage)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
    
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def start_camera(self):
        self.th_input.signal_stop()
        self.th_input.change_input2camera()
        self.th_input.signal_start()
        self.th_input.start()

    def start_video(self):
        self.th_input.signal_stop()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video",
            QDir.homePath())
        if file_name != '':
            self.th_input.set_file_name(file_name)
            self.th_input.change_input2video()
            self.th_input.signal_start()
            self.th_input.start()
