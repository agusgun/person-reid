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

from app.thread.input_thread import InputThread
from app.thread.detection_tracking_thread import DetectionTrackingThread

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

class InputTab(QWidget):
    def __init__(self, parent=None):
        super(InputTab, self).__init__(parent)

        self.input_widget = InputWidget()
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.input_widget)
        self.setLayout(layout)

class DetectionTrackingTab(QWidget):
    def __init__(self, input_thread, parent=None):
        super(DetectionTrackingTab, self).__init__(parent)
        self.input_thread = input_thread

        self.label = QLabel()
        self.label.resize(800, 600)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)

        self.th_detection_tracking = DetectionTrackingThread(input_thread)
        self.th_detection_tracking.changePixmap.connect(self.setImage)
        
    @pyqtSlot(QImage)
    def setImage(self, image):
        self.label.setPixmap(QPixmap.fromImage(image))


class TabWidget(QWidget):
    def __init__(self, parent=None):
        super(TabWidget, self).__init__(parent)

        self.tabs = QTabWidget()
        self.input_tab = InputTab()
        self.detection_tracking_tab = DetectionTrackingTab(self.input_tab.input_widget.th_input)
        self.reid_tab = QWidget()
        self.tabs.addTab(self.input_tab, "Input")
        self.tabs.addTab(self.detection_tracking_tab, "Detection and Tracking")
        self.tabs.addTab(self.reid_tab, "Re-identification")

        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tabs)
        self.setLayout(layout)


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

        self.tabs = TabWidget()

        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        wid = QWidget(self)
        self.setCentralWidget(wid)
        wid.setLayout(layout)

    def use_camera(self):
        self.tabs.input_tab.input_widget.start_camera()
        self.tabs.detection_tracking_tab.th_detection_tracking.signal_start()

    def open_file(self):
        self.tabs.input_tab.input_widget.start_video()
        self.tabs.detection_tracking_tab.th_detection_tracking.signal_start()

    def exit_call(self):
        sys.exit(app.exec_())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    player = MainWindow()
    player.resize(640, 480)
    player.show()
    sys.exit(app.exec_())