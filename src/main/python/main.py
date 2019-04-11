from app.thread.input_thread import InputThread
from app.thread.detection_tracking_thread import DetectionTrackingThread
from app.widget.tab_widget import TabWidget

from fbs_runtime.application_context import ApplicationContext
from PyQt5.QtWidgets import QAction, QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt5.QtGui import QIcon

import sys
import argparse
import os
import shutil

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)

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

class AppContext(ApplicationContext):           # 1. Subclass ApplicationContext
    def run(self):                              # 2. Implement run()
        version = self.build_settings['version']
        window = MainWindow()
        window.setWindowTitle("Person Re-ID v" + version)
        window.resize(800, 600)
        window.show()
        return self.app.exec_()                 # 3. End run() with this line

if __name__ == '__main__':
    curr_dir_path = os.path.dirname(__file__)
    keyframe_dir_path = os.path.join(curr_dir_path, 'app/keyframe_output')
    frame_dir_path = os.path.join(curr_dir_path, 'app/frame_output')
    
    keyframe_paths = [os.path.join(keyframe_dir_path, path) for path in os.listdir(keyframe_dir_path)]
    frame_paths = [os.path.join(frame_dir_path, path) for path in os.listdir(frame_dir_path)]
    
    for path in keyframe_paths:
        shutil.rmtree(path)
    for path in frame_paths:
        shutil.rmtree(path)

    appctxt = AppContext()                      # 4. Instantiate the subclass
    exit_code = appctxt.run()                   # 5. Invoke run()
    sys.exit(exit_code)