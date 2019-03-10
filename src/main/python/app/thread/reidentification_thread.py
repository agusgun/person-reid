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
        
    def run(self):
        while (True):
            if not self.keyframe_id_queue.empty():
                keyframe_id = self.keyframe_id_queue.get()

                print('Extract Keyframe', keyframe_id)
                # Extract face keyframe
                self.face_keyframe.generate_keyframes_from_frames(keyframe_id, self.person_frame_dir_path, self.face_keyframe_output_dir_path)
                
                # Train and predict face keyframe
                print('Reidentify', keyframe_id)
                if self.face_reidentification.is_not_trained():
                    print('First Train', keyframe_id)
                    self.face_reidentification.init_train_model(classifier='svm')
                    self.update_display_trigger.emit(self.face_reidentification.keyframe_person_dict[keyframe_id], keyframe_id)
                    # self.update_time_trigger.emit(self.face_reidentification.keyframe_person_dict[keyframe_id], keyframe_id)
                else:
                    print('Predict', keyframe_id)
                    predicted_proba = self.face_reidentification.predict_batch(keyframe_id, classifier='svm', return_proba=True)
                    if predicted_proba is None:
                        print('No Face Found', keyframe_id)
                    else:
                        print('Face Found', keyframe_id)
                        confidence_arr = np.max(predicted_proba, axis=0) # column
                        print(confidence_arr, keyframe_id)
                        
                        # Check Confidence below threshold
                        is_confidence_below_threshold = True
                        for confidence in confidence_arr[1:]:
                            if confidence > 0.7:
                                is_confidence_below_threshold = False
                                break

                        if confidence_arr[0] > 0.7 or is_confidence_below_threshold: # new person
                            print('New Person Update', keyframe_id)
                            self.face_reidentification.update_model(keyframe_id, classifier='svm')
                            self.update_display_trigger.emit(self.face_reidentification.keyframe_person_dict[keyframe_id], keyframe_id)
                        else: # existing person
                            predicted = np.argmax(confidence_arr)
                            print('Existing Person', keyframe_id, predicted)

    @pyqtSlot(int)
    def set_person_id(self, value):
        self.keyframe_id_queue.put(value)