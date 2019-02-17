import sys
import scipy.io as sio
from PyQt5 import QtGui, QtCore, QtWidgets
import cv2


class VideoCapture(QtWidgets.QWidget):
    def __init__(self, filename, parent):
        super(QtWidgets.QWidget, self).__init__()
        self.cap = cv2.VideoCapture(str(filename))
        self.video_frame = QtWidgets.QLabel()
        parent.layout.addWidget(self.video_frame)

    def nextFrameSlot(self):
        ret, frame = self.cap.read()
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2RGB)
        img = QtWidgets.QImage(frame, frame.shape[1], frame.shape[0], QtWidgets.QImage.Format_RGB888)
        pix = QtWidgets.QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0/30)

    def pause(self):
        self.timer.stop()

    def deleteLater(self):
        self.cap.release()
        super(QtWidgets.QWidget, self).deleteLater()


class VideoDisplayWidget(QtWidgets.QWidget):
    def __init__(self,parent):
        super(VideoDisplayWidget, self).__init__(parent)

        self.layout = QtWidgets.QFormLayout(self)

        self.startButton = QtWidgets.QPushButton('Start', parent)
        self.startButton.clicked.connect(parent.startCapture)
        self.startButton.setFixedWidth(50)
        self.pauseButton = QtWidgets.QPushButton('Pause', parent)
        self.pauseButton.setFixedWidth(50)
        self.layout.addRow(self.startButton, self.pauseButton)

        self.setLayout(self.layout)


class ControlWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(ControlWindow, self).__init__()
        self.setGeometry(50, 50, 800, 600)
        self.setWindowTitle("PyTrack")

        self.capture = None

        self.matPosFileName = None
        self.videoFileName = None
        self.positionData = None
        self.updatedPositionData  = {'red_x':[], 'red_y':[], 'green_x':[], 'green_y': [], 'distance': []}
        self.updatedMatPosFileName = None

        self.isVideoFileLoaded = False
        self.isPositionFileLoaded = False

        self.quitAction = QtWidgets.QAction("&Exit", self)
        self.quitAction.setShortcut("Ctrl+Q")
        self.quitAction.setStatusTip('Close The App')
        self.quitAction.triggered.connect(self.closeApplication)

        self.openMatFile = QtWidgets.QAction("&Open Position File", self)
        self.openMatFile.setShortcut("Ctrl+Shift+T")
        self.openMatFile.setStatusTip('Open .mat File')
        self.openMatFile.triggered.connect(self.loadPosMatFile)

        self.openVideoFile = QtWidgets.QAction("&Open Video File", self)
        self.openVideoFile.setShortcut("Ctrl+Shift+V")
        self.openVideoFile.setStatusTip('Open .h264 File')
        self.openVideoFile.triggered.connect(self.loadVideoFile)

        self.mainMenu = self.menuBar()
        self.fileMenu = self.mainMenu.addMenu('&File')
        self.fileMenu.addAction(self.openMatFile)
        self.fileMenu.addAction(self.openVideoFile)
        self.fileMenu.addAction(self.quitAction)

        self.videoDisplayWidget = VideoDisplayWidget(self)
        self.setCentralWidget(self.videoDisplayWidget)

    def startCapture(self):
        if not self.capture and self.isPositionFileLoaded and self.isVideoFileLoaded:
            self.capture = VideoCapture(self.videoFileName, self.videoDisplayWidget)
            self.videoDisplayWidget.pauseButton.clicked.connect(self.capture.pause)
        self.capture.start()

    def endCapture(self):
        self.capture.deleteLater()
        self.capture = None

    def loadPosMatFile(self):
        try:
            self.matPosFileName = str(QtWidgets.QFileDialog.getOpenFileName(self, 'Select .mat position File'))
            self.positionData = sio.loadmat(self.matPosFileName)
            self.isPositionFileLoaded = True
        except:
            print("Please select a .mat file")

    def loadVideoFile(self):
        try:
            self.videoFileName = QtWidgets.QFileDialog.getOpenFileName(self, 'Select .h264 Video File')
            self.isVideoFileLoaded = True
        except:
            print ("Please select a .h264 file")

    def closeApplication(self):
        choice = QtGui.QMessageBox.question(self, 'Message','Do you really want to exit?',QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if choice == QtGui.QMessageBox.Yes:
            print("Closing....")
            sys.exit()
        else:
            pass


if __name__ == '__main__':
    import sys
    app = QtWidgets.QApplication(sys.argv)
    window = ControlWindow()
    window.show()
    sys.exit(app.exec_())