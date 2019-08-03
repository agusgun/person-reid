from app.deep_sort.detection import Detection as ddet
from app.reidentification.face_reidentification import FaceReidentification
from app.keyframe.face_detector import FaceDetector
import cv2 as cv
import os
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
from PyQt5.QtGui import QImage
import time

class DetectionReidentificationThread(QThread):
    changePixmap = pyqtSignal(QImage)
    update_display_trigger = pyqtSignal(int, int, str)
    
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
        self.face_reidentification = FaceReidentification()
        self.THRESHOLD = 80
        self.FACE_IMG_SIZE = (60, 60)
        curr_dir_path = os.path.dirname(__file__)
        self.face_keyframe_output_dir_path = os.path.join(curr_dir_path, '../keyframe_output/')
        
    def run(self):
        while True and self.is_running:
            ret, frame = self.input_thread.get_capture()
            if ret:
                start_time = time.time()

                # Detection Stuff Here
                face_landmarks = self.face_detector.detect_all_and_extract_landmark(frame)
                for face_landmark in face_landmarks:
                    x, y, w, h = face_landmark['box']
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if h > 60:
                        cropped_img = frame[y:y+h, x:x+w]
                        cv.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255,0,0), 2)
                        
                        # Extract feature
                        feature = self.face_reidentification.extract_feature_from_image(cropped_img)
                        self.keyframe_id_counter += 1
                        keyframe_image_path = os.path.join(self.face_keyframe_output_dir_path, str(self.keyframe_id_counter) + '.png')
                        cropped_img = cv.resize(cropped_img, self.FACE_IMG_SIZE, interpolation=cv.INTER_AREA)
                        cv.imwrite(keyframe_image_path, cropped_img)

                        # Predict
                        prediction_match = self.face_reidentification.predict_and_find_match_single(self.image_representation_database, 
                            self.image_representation_paths, self.image_representation_label, feature, self.THRESHOLD)
                        minimum_label, minimum_path = prediction_match

                        # Add to galery
                        self.image_representation_database.append(feature)
                        self.image_representation_paths.append(keyframe_image_path)
                        if minimum_label is None: # Predicted as new
                            self.label_counter += 1
                            self.image_representation_label.append(self.label_counter)
                            self.update_display_trigger.emit(self.label_counter, self.keyframe_id_counter, minimum_path)
                        else: # Predicted as old
                            self.image_representation_label.append(minimum_label)
                            self.update_display_trigger.emit(minimum_label, self.keyframe_id_counter, minimum_path)
                    else:
                        cv.rectangle(frame, (int(x), int(y)), (int(x+w), int(y+h)), (255,255,255), 2)


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
