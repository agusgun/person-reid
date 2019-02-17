# PyQt5 Video player
#!/usr/bin/env python

from PyQt5.QtCore import QDir, Qt, QUrl, QThread, pyqtSignal, pyqtSlot
from PyQt5.QtMultimedia import QMediaContent, QMediaPlayer
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QFileDialog, QHBoxLayout, QLabel,
        QPushButton, QSizePolicy, QSlider, QStyle, QVBoxLayout, QWidget, QTabWidget)
from PyQt5.QtWidgets import QMainWindow,QWidget, QPushButton, QAction
from PyQt5.QtGui import QIcon, QPixmap, QImage
import sys
import os
import cv2 as cv


# Video Thread
class CameraThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def run(self):
        cap = cv.VideoCapture(0)
        while True and self.is_running:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
        cap.release()

    def signal_start(self):
        self.is_running = True

    def signal_stop(self):
        self.is_running = False

class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self):
        super().__init__()
        self.is_running = True

    def set_file_name(self, file_name):
        self.file_name = file_name
    
    def run(self):
        cap = cv.VideoCapture(self.file_name)
        while True and self.is_running:
            ret, frame = cap.read()
            if ret:
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            # Add Delay
            self.msleep(30)
        cap.release()

    def signal_start(self):
        self.is_running = True

    def signal_stop(self):
        self.is_running = False

class MainWindow(QMainWindow):
    
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setWindowTitle("Person Re-ID")

        # Menu Bar
        use_camera_action = QAction(QIcon('camera.png'), '&Use Camera', self)
        use_camera_action.setShortcut('Ctrl+E')
        use_camera_action.setStatusTip('Use camera')
        use_camera_action.triggered.connect(self.use_camera)

        open_action = QAction(QIcon('open.png'), '&Open', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('Open video')
        open_action.triggered.connect(self.open_file)

        exit_action = QAction(QIcon('exit.png'), '&Exit', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('Exit application')
        exit_action.triggered.connect(self.exit_call)

        menu_bar = self.menuBar()
        file_menu = menu_bar.addMenu('&File')
        file_menu.addAction(use_camera_action)
        file_menu.addAction(open_action)
        file_menu.addAction(exit_action)

        # Tab
        self.tabs = QTabWidget()
        self.tab1 = QWidget()
        self.tab2 = QWidget()
        self.tab3 = QWidget()
        self.tabs.addTab(self.tab1, "Input")
        self.tabs.addTab(self.tab2, "Detection and Tracking")
        self.tabs.addTab(self.tab3, "Re-identification")


        # Central Widget
        layout = QVBoxLayout()

        self.label = QLabel()
        self.label.resize(640, 480)

        self.th_camera = CameraThread()
        self.th_camera.changePixmap.connect(self.setImage)
        self.th_video = VideoThread()
        self.th_video.changePixmap.connect(self.setImage)
        
        layout.addWidget(self.tabs)
        layout.addWidget(self.label)

        # Apply Layout
        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)

    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))

    def use_camera(self):
        self.th_video.signal_stop()
        self.th_camera.signal_start()
        self.th_camera.start()

    def open_file(self):
        self.th_camera.signal_stop()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open Video",
            QDir.homePath())
        if file_name != '':
            self.th_video.set_file_name(file_name)
            self.th_video.signal_start()
            self.th_video.start()

    def exit_call(self):
        sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())