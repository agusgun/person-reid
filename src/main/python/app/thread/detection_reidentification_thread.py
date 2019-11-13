from app.deep_sort.detection import Detection as ddet
from app.reidentification.face_reidentification import FaceReidentification
from app.keyframe.face_detector import FaceDetector
from app.sort.sort import Sort

import cv2 as cv
import numpy as np
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread, QVariant
from PyQt5.QtGui import QImage
import time

def normalize_keypoint(point, x, y):
    return (point[0] - x, point[1] - y)

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

class DetectionReidentificationThread(QThread):
    changePixmap = pyqtSignal(QImage)
    update_display_trigger = pyqtSignal(int, int, QVariant)
    
    def __init__(self, input_thread):
        super().__init__()
        self.is_running = True
        self.input_thread = input_thread
        
        # Detection
        self.face_detector = FaceDetector(detector='mtcnn')

        # Reidentification
        self.keyframe_id_counter = 0
        self.label_counter = 0
        self.image_representation_database = []
        self.image_representation_label = []
        self.image_representation_paths = []
        self.tracker = Sort()
        self.face_reidentification = FaceReidentification()
        self.THRESHOLD = 80
        self.FACE_IMG_SIZE = (60, 60)
        curr_dir_path = os.path.dirname(__file__)
        self.face_keyframe_output_dir_path = os.path.join(curr_dir_path, '../keyframe_output/')
        self.person_iterator_dict = dict()
        self.counter = 0

    def _find_majority_prediction_match(self, prediction_matches):
        map = {}
        maximum = (None, 0)
        for prediction_match in prediction_matches:
            prediction = prediction_match[0]
            if prediction in map: 
                map[prediction] += 1
            else: 
                map[prediction] = 1

            if map[prediction] > maximum[1]:
                maximum = (prediction, map[prediction])
        return maximum

    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                start_time = time.time()

                # Detection Stuff Here
                face_landmarks = self.face_detector.detect_all_and_extract_landmark(frame)
                detections = []
                facial_landmarks = []
                for face_landmark in face_landmarks:
                    x, y, w, h = face_landmark['box']
                    confidence_score = face_landmark['confidence']
                    detections.append([x, y, x+w, y+h, confidence_score])                    
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    
                    cropped_img = frame[y:y+h, x:x+w].copy()
                    keypoints_landmark = face_landmark['keypoints']
                    left_eye_position = np.array(list(normalize_keypoint(keypoints_landmark['left_eye'], x, y)))
                    right_eye_position = np.array(list(normalize_keypoint(keypoints_landmark['right_eye'], x, y)))
                    nose_position = np.array(list(normalize_keypoint(keypoints_landmark['nose'], x, y)))
                    mouth_left_position = np.array(list(normalize_keypoint(keypoints_landmark['mouth_left'], x, y)))
                    mouth_right_position = np.array(list(normalize_keypoint(keypoints_landmark['mouth_right'], x, y)))
                    
                    facial_landmarks.append([left_eye_position, right_eye_position, nose_position, mouth_left_position, mouth_right_position])
                
                detections = np.array(detections)
                # print(detections)
                img_size = np.asarray(frame.shape)[0:2]
                dir_path = "test"
                detection_interval = 1
                tracker_result = self.tracker.update(detections)
                for track in tracker_result:
                    bbox_track = track.astype(np.int32)
                    track_id = int(track[4])
                    for idx_det, det in enumerate(detections):
                        bbox_det = det[:4]
                        if bb_intersection_over_union(bbox_det, bbox_track) > 0.5:
                            facial_landmark = facial_landmarks[idx_det]
                            left_right_eye_distance = np.linalg.norm(facial_landmark[0] - facial_landmark[1])
                            left_eye_mouth_distance = np.linalg.norm(facial_landmark[0] - facial_landmark[3])
                            distance_ratio = left_right_eye_distance / left_eye_mouth_distance

                            dX = facial_landmark[1][0] - facial_landmark[0][0]
                            dY = facial_landmark[1][1] - facial_landmark[0][1]
                            angle = np.degrees(np.arctan2(dY, dX))

                            cropped_img = frame[int(bbox_det[1]): int(bbox_det[3]), int(bbox_det[0]): int(bbox_det[2])]
                            if distance_ratio > 0.85 and angle <= 10 and angle >= -10 and self.person_iterator_dict.get(track_id, 0) < 2: # heuristic
                                if track_id in self.person_iterator_dict:
                                    self.person_iterator_dict[track_id] += 1
                                else:
                                    self.person_iterator_dict[track_id] = 0

                                person_id_dir_path = os.path.join(self.face_keyframe_output_dir_path, str(track_id))
                                if not os.path.exists(person_id_dir_path):
                                    os.makedirs(person_id_dir_path)
                                frame_output_file_path = os.path.join(person_id_dir_path, str(self.person_iterator_dict[track_id]) + '.png')                                                
                                cropped_img = cv.resize(cropped_img, self.FACE_IMG_SIZE, interpolation=cv.INTER_AREA)
                                
                                cv.imwrite(os.path.join(frame_output_file_path), cropped_img)
                                # Re-identification
                                if self.person_iterator_dict.get(track_id, 0) == 2: # TODO: add condition if leave an area
                                    features = self.face_reidentification.extract_feature_from_keyframes(self.face_keyframe_output_dir_path, track_id)
                                    prediction_matches = self.face_reidentification.predict_and_find_match_batch(self.image_representation_database, self.image_representation_paths, self.image_representation_label, features, self.THRESHOLD)
                                    current_face_keyframe_dir_path = os.path.join(self.face_keyframe_output_dir_path, str(track_id))
                                    image_paths = [os.path.join(current_face_keyframe_dir_path, path) for path in os.listdir(current_face_keyframe_dir_path)]
                                    majority = self._find_majority_prediction_match(prediction_matches)
                                    print('Majority', majority)
                                    if majority[1] != 0: # no keyframe found
                                        if majority[0] == None: # new person
                                            print('New Person')
                                            self.counter += 1
                                            self.image_representation_database.append(features)
                                            self.image_representation_label.append(self.counter)
                                            self.image_representation_paths.append(image_paths)
                                            self.update_display_trigger.emit(self.counter, track_id, [prediction[1] for prediction in prediction_matches])
                                        else: # existing person
                                            print('Existing Person', majority[0])
                                            self.image_representation_database.append(features)
                                            self.image_representation_label.append(majority[0])
                                            self.image_representation_paths.append(image_paths)
                                            self.update_display_trigger.emit(self.counter, track_id, [prediction[1] for prediction in prediction_matches])

                                cv.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 255, 0), 2)
                            else:        
                                cv.rectangle(frame, (int(track[0]), int(track[1])), (int(track[2]), int(track[3])), (0, 0, 255), 2)
                    
                end_time = time.time()
                # Put efficiency information
                label = 'Inference time: %.2f ms' % ((end_time - start_time)*1000)
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
