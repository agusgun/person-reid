from app.deep_sort.detection import Detection as ddet
from app.detection_tracking.tracking import Tracking
import cv2 as cv
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage
import time

import platform
system_platform = platform.system()

if bool(os.environ['USE_GPU']):
    USE_GPU = True
else:
    USE_GPU = False     

if system_platform == 'Linux' and USE_GPU:
    from app.detection_tracking.detection import Detection
    from pydarknet import Image
else:
    from app.detection_tracking.detection_opencv import DetectionOpenCV
print(system_platform)

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
    change_person_id = pyqtSignal(int)
    
    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        if system_platform == 'Linux' and USE_GPU:
            self.detection = Detection()
        else:
            self.detection = DetectionOpenCV()
        self.tracking = Tracking()
        self.person_iterator_dict = dict()
        
    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                start_time = time.time()

                # Detection Stuff Here
                if system_platform == 'Linux' and USE_GPU:
                    dark_frame = Image(frame)
                    outs = self.detection.net.detect(dark_frame, thresh=self.detection.confThreshold, nms=self.detection.nmsThreshold)
                else:
                    blob = cv.dnn.blobFromImage(frame, 1/255, (self.detection.inpWidth, self.detection.inpHeight), [0,0,0], 1, crop=False)
                    self.detection.net.setInput(blob)
                    outs = self.detection.net.forward(self.detection.getOutputsNames(self.detection.net))

                # Remove the bounding boxes with low confidence
                boxes_for_tracking = self.detection.postprocess(frame, outs)
                features = self.tracking.encoder(frame, boxes_for_tracking)
                detections = [ddet(bbox, 1.0, feature) for bbox, feature in zip(boxes_for_tracking, features)]
                self.tracking.tracker.predict()
                self.tracking.tracker.update(detections)

                for track in self.tracking.tracker.tracks:
                    # Trigger for keyframe extraction
                    if track.time_since_update == self.tracking.max_age:
                        self.change_person_id.emit(track.track_id)
                        
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    
                    for idx_det, det in enumerate(detections):
                        bbox_detection = det.to_tlbr()
                        bbox_tracking = track.to_tlbr()
                        if system_platform != 'Linux' or not(USE_GPU):
                            self.detection.drawPred(frame, 'Human', 101, int(bbox_detection[0]), int(bbox_detection[1]), int(bbox_detection[2]), int(bbox_detection[3]))
                        if bb_intersection_over_union(bbox_detection, bbox_tracking) > 0.5:
                            del detections[idx_det]
                            # Check if the left of tracking position is still in the frame (if not will crop empty image) TODO: better handler
                            if (bbox_tracking[0] >= 0 and bbox_tracking[1] >= 0):
                                if track.track_id in self.person_iterator_dict:
                                    self.person_iterator_dict[track.track_id] += 1
                                else:
                                    self.person_iterator_dict[track.track_id] = 0

                                cropped_img = frame[int(bbox_tracking[1]): int(bbox_tracking[3]), int(bbox_tracking[0]): int(bbox_tracking[2])]
                                curr_dir_path = os.path.abspath(os.path.dirname(__file__))
                                output_dir_path = os.path.join(curr_dir_path, '../frame_output/')
                                person_id_dir_path = os.path.join(output_dir_path, str(track.track_id))
                                if not os.path.exists(person_id_dir_path):
                                    os.makedirs(person_id_dir_path)
                                frame_output_file_path = os.path.join(person_id_dir_path, str(self.person_iterator_dict[track.track_id]) + '.png')        
                                
                                cv.imwrite(os.path.join(frame_output_file_path), cropped_img)
                                
                            cv.rectangle(frame, (int(bbox_detection[0]), int(bbox_detection[1])), (int(bbox_detection[2]), int(bbox_detection[3])), (255,255,255), 2)
                            cv.putText(frame, str(track.track_id),(int(bbox_detection[0]), int(bbox_detection[1])), 0, 5e-3 * 200, (0,255,0), 2)

                end_time = time.time()
                # Put efficiency information
                label = 'Inference time: %.2f ms' % ((end_time - start_time)*1000)
                cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
                # Convert frame to display to PyQt
                rgbImage = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                convertToQtFormat = QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QImage.Format_RGB888)
                p = convertToQtFormat.scaled(800, 600, Qt.KeepAspectRatio)
                self.changePixmap.emit(p)
            else: # when no frame extracted but re-identification still in progress
                for track in self.tracking.tracker.tracks:
                    if track.time_since_update == self.tracking.max_age:
                        self.change_person_id.emit(track.track_id)
                
    def signal_start(self):
        self.is_running = True
        self.start()

    def signal_stop(self):
        self.is_running = False
