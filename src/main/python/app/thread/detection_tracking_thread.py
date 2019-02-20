from app.deep_sort.detection import Detection as ddet
from app.detection_tracking.detection import Detection
from app.detection_tracking.tracking import Tracking
import cv2 as cv
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage

def bb_intersection_over_union(boxA, boxB):
	# determine the (x, y)-coordinates of the intersection rectangle
	xA = max(boxA[0], boxB[0])
	yA = max(boxA[1], boxB[1])
	xB = min(boxA[2], boxB[2])
	yB = min(boxA[3], boxB[3])
 
	# compute the area of intersection rectangle
	interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
 
	# compute the area of both the prediction and ground-truth
	# rectangles
	boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
	boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
	# compute the intersection over union by taking the intersection
	# area and dividing it by the sum of prediction + ground-truth
	# areas - the interesection area
	iou = interArea / float(boxAArea + boxBArea - interArea)
 
	# return the intersection over union value
	return iou

class DetectionTrackingThread(QThread):
    changePixmap = pyqtSignal(QImage)

    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        self.detection = Detection()
        self.tracking = Tracking()
        self.person_iterator_dict = dict()

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
                    
                    for idx_det, det in enumerate(detections):
                        bbox_detection = det.to_tlbr()
                        bbox_tracking = track.to_tlbr()

                        if bb_intersection_over_union(bbox_detection, bbox_tracking) > 0.5:
                            del detections[idx_det]

                            cv.rectangle(frame, (int(bbox_detection[0]), int(bbox_detection[1])), (int(bbox_detection[2]), int(bbox_detection[3])), (255,255,255), 2)
                            cv.putText(frame, str(track.track_id),(int(bbox_detection[0]), int(bbox_detection[1])), 0, 5e-3 * 200, (0,255,0), 2)

                # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
                t, _ = self.detection.net.getPerfProfile()
                label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                print(label)
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
