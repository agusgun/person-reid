from app.deep_sort.detection import Detection as ddet
from app.detection_tracking.detection import Detection
from app.detection_tracking.tracking import Tracking
import cv2 as cv
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage

class DetectionTrackingThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        self.detection = Detection()
        self.tracking = Tracking()

    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                # Detection Stuff Here
                # Create a 4D blob from a frame.
                blob = cv.dnn.blobFromImage(frame, 1/255, (self.detection.inpWidth, self.detection.inpHeight), [0,0,0], 1, crop=False)
                # Sets the input to the network
                self.detection.net.setInput(blob)
                # Runs the forward pass to get output of the output layers
                outs = self.detection.net.forward(self.detection.getOutputsNames(self.detection.net))
                # Remove the bounding boxes with low confidence
                boxes_for_tracking = self.detection.postprocess(frame, outs)
                features = self.tracking.encoder(frame, boxes_for_tracking)
                detections = [ddet(bbox, 1.0, feature) for bbox, feature in zip(boxes_for_tracking, features)]

                self.tracking.tracker.predict()
                self.tracking.tracker.update(detections)

                for track in self.tracking.tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    bbox = track.to_tlbr()
                    cv.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
                    cv.putText(frame, str(track.track_id),(int(bbox[0]), int(bbox[1])),0, 5e-3 * 200, (0,255,0),2)

                for det in detections:
                    bbox = det.to_tlbr()
                    cv.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,0,0), 2)


                # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = self.detection.net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                
                # Convert frame to display to PyQt
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
                
    def signal_start(self):
        self.is_running = True
        self.start()

    def signal_stop(self):
        self.is_running = False
