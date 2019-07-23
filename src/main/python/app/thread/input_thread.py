import cv2 as cv
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage

class InputThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, video=False):
        super().__init__()
        self.is_running = True
        self.ret = None
        self.cap = None

    def run(self):
        while True and self.is_running:
            self.ret, self.frame = self.cap.read()
            if self.ret:
                rgbImage = cv.cvtColor(self.frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                self.msleep(30)
        print('Cap Rel')
        self.cap.release()

    def set_file_name(self, file_name):
        self.file_name = file_name
    
    def get_capture(self):
        if self.ret:
            return self.ret, self.frame
        else:
            return None, None

    def change_input2video(self):
        self.cap = cv.VideoCapture(self.file_name)
        print(self.cap.get(cv.CAP_PROP_FPS), self.cap.get(cv.CAP_PROP_FRAME_COUNT))

    def change_input2camera(self):
        self.cap = cv.VideoCapture(0)
        # print(self.cap.get(cv.CAP_PROP_FPS), self.cap.get(cv.CAP_PROP_FRAME_COUNT))

    def signal_start(self):
        self.is_running = True

    def signal_stop(self):
        self.is_running = False