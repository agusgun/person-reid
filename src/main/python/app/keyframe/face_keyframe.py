import cv2 as cv
import dlib
from imutils import face_utils
import numpy as np
import os
from scipy.spatial import distance
from .face_detector import FaceDetector
import matplotlib.pyplot as plt

class FaceKeyframe: 
    def __init__(self):
        self.face_detector = FaceDetector(detector='mtcnn')
        self.IMG_SIZE = (60, 60)

    def _normalize_keypoints(self, point, x, y):
        return (point[0] - x, point[1] - y)
    
    def face_keyframe_check(self, filename, lower_threshold, higher_threshold):
        img = cv.imread(filename)
        if img is not None:
            img = cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), self.IMG_SIZE, interpolation=cv.INTER_AREA)
            face_and_landmark = self.face_detector.detect_and_extract_landmark(img)
            if face_and_landmark is not None:
                x, y, w, h = face_and_landmark['box']
                if x < 0:
                    x = 0
                if y < 0:
                    y = 0        
                cropped_img = img[y:y+h, x:x+w]
                
                keypoints = face_and_landmark['keypoints']
                # Normalize position
                le_pos = self._normalize_keypoints(keypoints['left_eye'], x, y) 
                re_pos =  self._normalize_keypoints(keypoints['right_eye'], x, y)
                nose_pos =  self._normalize_keypoints(keypoints['nose'], x, y)
                ml_pos =  self._normalize_keypoints(keypoints['mouth_left'], x, y)
                mr_pos =  self._normalize_keypoints(keypoints['mouth_right'], x, y)

                dX = re_pos[0] - le_pos[0]
                dY = re_pos[1] - le_pos[1]
                angle = np.degrees(np.arctan2(dY, dX))
                if angle >= lower_threshold and angle <= higher_threshold:
                    return (cropped_img, angle)
                else:
                    return None
            else:
                return None
        else:
            return None
        
    def generate_keyframes_from_frames(self, person_id, input_dir_path, output_dir_path):
        person_input_dir_path = os.path.join(input_dir_path, str(person_id))
        if os.path.exists(person_input_dir_path):
            file_paths = os.listdir(person_input_dir_path)
            file_paths = [os.path.join(person_input_dir_path, file_path) for file_path in file_paths]
            counter = 0
            print("generate keyframe from keyframe_id = ", person_id)
            MAX = 100

            face_images = []
            LOWER_THRESHOLD = -5
            HIGHER_THRESHOLD = 5
            for file_path in file_paths[:MAX]:
                face_img = self.face_keyframe_check(file_path, LOWER_THRESHOLD, HIGHER_THRESHOLD)
                if face_img is not None:
                    face_images.append(face_img)
            
            keyframe_output_dir_path = os.path.join(output_dir_path, str(person_id))
            if not os.path.exists(keyframe_output_dir_path):
                os.makedirs(keyframe_output_dir_path)

            face_images = sorted(face_images, key=lambda x: x[1])
            increment = 1
            NUMBER_OF_KEYFRAME = 10
            if len(face_images) > NUMBER_OF_KEYFRAME:
                increment = len(face_images) // NUMBER_OF_KEYFRAME
            
            counter = 0
            for face_img in face_images[::increment]:
                img, angle = face_img
                keyframe_output_path = os.path.join(keyframe_output_dir_path, str(counter) + '.png')
                img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
                cv.imwrite(keyframe_output_path, img)
                counter += 1
                if counter == NUMBER_OF_KEYFRAME:
                    break