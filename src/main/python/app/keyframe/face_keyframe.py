import cv2 as cv
import dlib
from imutils import face_utils
import numpy as np
import os
from scipy.spatial import distance

class FaceKeyframe: 
    def __init__(self):
        model_dir_path = os.path.join(os.path.dirname(__file__) + '../model_data/')
        cascade_classifier_path = os.path.join(model_dir_path + 'haarcascade_frontalface_alt.xml')
        facial_landmark_extractor_path = os.path.join(model_dir_path + 'shape_predictor_68_face_landmarks.dat')

        self.cascade_classifier = cv.CascadeClassifier(cascade_classifier_path)
        self.facial_landmark_extractor = dlib.shape_predictor(facial_landmark_extractor_path)

        self.FACIAL_LANDMARKS_CORNERS_IDXS = {
            'jaw': (0, 16),
            'right_eyebrow': (17, 21),
            'left_eyebrow': (22, 26),
            'nose': (27, 33), # From Top to Bot
            'right_eye': (36, 39), 
            'left_eye': (42, 45),
            'mouth': (48, 54)
        }

    def detect_faces(self, gray_img, scale_factor=1.1, min_neighbor=5):
        face_bboxes_list = self.cascade_classifier.detectMultiScale(gray_img, scaleFactor=scale_factor, minNeighbors=min_neighbor)
        return face_bboxes_list

    def facial_landmark_extraction(self, gray_img, face_bbox_position):
        x, y, w, h = face_bbox_position

        bbox_rectangle = dlib.rectangle(x, y, x+w, y+h)

        facial_landmark_position = self.facial_landmark_extractor(gray_img, bbox_rectangle)
        facial_landmark_position = face_utils.shape_to_np(facial_landmark_position)

        return facial_landmark_position

    def landmark_size_extraction(self, facial_landmark_position):
        landmark = facial_landmark_position 
        jaw_idx_corner1, jaw_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['jaw']
        jaw_size = distance.euclidean(landmark[jaw_idx_corner1], landmark[jaw_idx_corner2])

        right_eyebrow_idx_corner1, right_eyebrow_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['right_eyebrow']
        right_eyebrow_size = distance.euclidean(landmark[right_eyebrow_idx_corner1], landmark[right_eyebrow_idx_corner2])

        left_eyebrow_idx_corner1, left_eyebrow_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['left_eyebrow']
        left_eyebrow_size = distance.euclidean(landmark[left_eyebrow_idx_corner1], landmark[left_eyebrow_idx_corner2])

        nose_idx_corner1, nose_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['nose']
        nose_size = distance.euclidean(landmark[nose_idx_corner1], landmark[nose_idx_corner2])

        right_eye_idx_corner1, right_eye_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['right_eye']
        right_eye_size = distance.euclidean(landmark[right_eye_idx_corner1], landmark[right_eye_idx_corner2])

        left_eye_idx_corner1, left_eye_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['left_eye']
        left_eye_size = distance.euclidean(landmark[left_eye_idx_corner1], landmark[left_eye_idx_corner2])

        mouth_idx_corner1, mouth_idx_corner2 = self.FACIAL_LANDMARKS_CORNERS_IDXS['mouth']
        mouth_size = distance.euclidean(landmark[mouth_idx_corner1], landmark[mouth_idx_corner2])

        return (jaw_size, right_eyebrow_size, left_eyebrow_size, nose_size, right_eye_size, left_eye_size, mouth_size)

    def face_keyframe_check(self, filename):
        gray_img = cv.imread(filename, cv.IMREAD_GRAYSCALE)

        bbox_of_faces = self.detect_faces(gray_img)
        for x, y, w, h in bbox_of_faces:
            face_img = gray_img[y:y+h, x:x+w]
            face_resized_dimension = (60, 60)
            face_img = cv.resize(face_img, face_resized_dimension, cv.INTER_AREA)

            facial_landmark_position = self.facial_landmark_extraction(face_img, (0, 0, face_resized_dimension[0], face_resized_dimension[1]))

            landmark_size = self.landmark_size_extraction(facial_landmark_position)
            if landmark_size != None:
                jaw_size, reb_size, leb_size, nose_size, re_size, le_size, mouth_size = landmark_size

                # Edit the condition here
                if (jaw_size > 50 and nose_size > 15):
                    return face_img
            else:
                return None

    def generate_keyframes_from_frames(self, person_id, input_path_list, output_dir_path):
        counter = 0
        for file_path in input_path_list:
            face_img = self.face_keyframe_check(file_path)
            if face_img != None:
                keyframe_output_path = 'K' + str(id) + '_' + counter + '.png'
                keyframe_output_path = os.path.join(output_dir_path, keyframe_output_path)
                cv.imwrite(keyframe_output_path, face_img)
                counter += 1