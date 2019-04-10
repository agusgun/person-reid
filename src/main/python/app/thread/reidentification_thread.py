from app.keyframe.face_keyframe import FaceKeyframe
from app.reidentification.face_reidentification import FaceReidentification
import numpy as np
import os
from PyQt5.QtCore import QThread, pyqtSlot, pyqtSignal
import queue

class ReidentificationThread(QThread):
    update_display_trigger = pyqtSignal(int, int)
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
        self.counter = 0
        self.THRESHOLD = 80
        
    def run(self):
        while (True):
            if not self.keyframe_id_queue.empty():
                keyframe_id = self.keyframe_id_queue.get()

                print('Extract Keyframe', keyframe_id)
                # Extract face keyframe
                self.face_keyframe.generate_keyframes_from_frames(keyframe_id, self.person_frame_dir_path, self.face_keyframe_output_dir_path)
                
                # Train and predict face keyframe
                print('Reidentify', keyframe_id)
                features = self.face_reidentification.extract_feature_from_keyframes(self.face_keyframe_output_dir_path, keyframe_id)
                predictions = self.face_reidentification.predict_batch(self.image_representation_database, self.image_representation_label, features, self.THRESHOLD)
                print('Predictions', predictions)
                majority = self._find_majority(predictions)
                print('Majority', majority)
                if majority[0] == None: # new person
                    print('New Person')
                    self.counter += 1
                    self.image_representation_database.append(features)
                    self.image_representation_label.append(self.counter)
                    self.update_display_trigger.emit(self.counter, keyframe_id)
                else: # existing person
                    print('Existing Person', majority[0])
                    self.image_representation_database.append(features)
                    self.image_representation_label.append(majority[0])
                    self.update_display_trigger.emit(majority[0], keyframe_id)


                

    def _find_majority(self, list_):
        map = {}
        maximum = (None, 0)
        for elmt in list_:
            if elmt in map: 
                map[elmt] += 1
            else: 
                map[elmt] = 1

            if map[elmt] > maximum[1]:
                maximum = (elmt, map[elmt])
        return maximum

    @pyqtSlot(int)
    def set_person_id(self, value):
        self.keyframe_id_queue.put(value)