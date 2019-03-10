from app.keyframe.face_keyframe import FaceKeyframe
import os
from PyQt5.QtCore import QThread, pyqtSlot

class ReidentificationThread(QThread):
    def __init__(self):
        super().__init__()
        self.is_triggered = False
        self.person_id = None
        curr_dir_path = os.path.dirname(__file__)
        self.person_frame_dir_path = os.path.join(curr_dir_path, '../frame_output/')
        self.face_keyframe = FaceKeyframe()
        self.face_keyframe_output_dir_path = os.path.join(curr_dir_path, '../keyframe_output/')
        

    def run(self):
        while (True):
            if self.is_triggered and self.person_id != None:
                # Extract keyframe here
                self.face_keyframe.generate_keyframes_from_frames(self.person_id, self.person_frame_dir_path, self.face_keyframe_output_dir_path)

                self.set_finished()


    @pyqtSlot(bool)
    def init_trigger(self, value):
        self.is_triggered = value

    @pyqtSlot(int)
    def set_person_id(self, value):
        self.person_id = value

    def set_finished(self):
        self.person_id = None
        self.is_triggered = False