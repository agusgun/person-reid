from app.keyframe.face_keyframe import FaceKeyframe
from app.reidentification.face_reidentification import FaceReidentification
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal, QVariant
import queue

class ReidentificationThread(QThread):
    update_display_trigger = pyqtSignal(int, int, QVariant)
    update_time_trigger = pyqtSignal(int, int)

    def __init__(self):
        super().__init__()
        self.is_triggered = False
        self.keyframe_id_queue = queue.Queue()
        curr_dir_path = os.path.dirname(__file__)
        self.person_frame_dir_path = os.path.join(curr_dir_path, '../frame_output/')
        self.face_keyframe = FaceKeyframe()
        self.face_keyframe_output_dir_path = os.path.join(curr_dir_path, '../keyframe_output/')

        self.face_reidentification = FaceReidentification()
        self.image_representation_database = []
        self.image_representation_label = []
        self.image_representation_paths = []
        self.counter = 0
        self.THRESHOLD = 80
        
    def run(self):
        while (True):
            if not self.keyframe_id_queue.empty():
                keyframe_id = self.keyframe_id_queue.get()

                print('Extract Keyframe', keyframe_id)
                # Extract face keyframe
                self.face_keyframe.generate_keyframes_from_frames(keyframe_id, self.person_frame_dir_path, self.face_keyframe_output_dir_path)
                
                # Predict face keyframe
                print('Reidentify', keyframe_id)
                features = self.face_reidentification.extract_feature_from_keyframes(self.face_keyframe_output_dir_path, keyframe_id)
                prediction_matches = self.face_reidentification.predict_and_find_match_batch(self.image_representation_database, self.image_representation_paths, self.image_representation_label, features, self.THRESHOLD)
                current_face_keyframe_dir_path = os.path.join(self.face_keyframe_output_dir_path, str(keyframe_id))
                image_paths = [os.path.join(current_face_keyframe_dir_path, path) for path in os.listdir(current_face_keyframe_dir_path)]
                
                majority = self._find_majority_prediction_match(prediction_matches)
                print('Majority', majority)
                if majority[0] == None: # new person
                    print('New Person')
                    self.counter += 1
                    self.image_representation_database.append(features)
                    self.image_representation_label.append(self.counter)
                    self.image_representation_paths.append(image_paths)
                    self.update_display_trigger.emit(self.counter, keyframe_id, [prediction[1] for prediction in prediction_matches])
                else: # existing person
                    print('Existing Person', majority[0])
                    self.image_representation_database.append(features)
                    self.image_representation_label.append(majority[0])
                    self.image_representation_paths.append(image_paths)
                    self.update_display_trigger.emit(majority[0], keyframe_id, [prediction[1] for prediction in prediction_matches])

    def _find_majority_prediction(self, predictions):
        map = {}
        maximum = (None, 0)
        for prediction in predictions:
            if prediction in map: 
                map[prediction] += 1
            else: 
                map[prediction] = 1

            if map[prediction] > maximum[1]:
                maximum = (prediction, map[prediction])
        return maximum

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

    @pyqtSlot(int)
    def set_person_id(self, value):
        self.keyframe_id_queue.put(value)