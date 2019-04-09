import os
import dlib
import cv2 as cv
from mtcnn.mtcnn import MTCNN
from imutils import face_utils


class FaceDetector:
    def __init__(self, detector='mtcnn'):
        curr_dir_path = os.path.dirname(__file__)
        model_dir_path = os.path.join(curr_dir_path, '../model_data/')
        CASCADE_CLASSIFIER_MODEL_PATH = os.path.join(model_dir_path, 'haarcascade_frontalface_alt.xml')
        FACIAL_LANDMARK_EXTRACTOR_PATH = os.path.join(model_dir_path + 'shape_predictor_68_face_landmarks.dat')
        self.detector_type = detector

        if detector == 'mtcnn':
            print('Load MTCNN Face Detector')
            self.detector = MTCNN()
            self.landmark_extractor = None
        elif detector == 'dlib_hog':
            print('Load HOG Face Detector')
            self.detector = dlib.get_frontal_face_detector()
            self.landmark_extractor = dlib.shape_predictor(FACIAL_LANDMARK_EXTRACTOR_PATH)
        elif detector == 'cascade':
            print('Load Cascade Face Detector')
            self.detector = cv.CascadeClassifier(CASCADE_CLASSIFIER_MODEL_PATH)
            self.landmark_extractor = dlib.shape_predictor(FACIAL_LANDMARK_EXTRACTOR_PATH)
        else:
            print('The detector not implemented')

    def detect_and_extract_landmark(self, person_img):
        '''
        Return first position of face found in format (x, y, w, h)->(left, top, width, height)
        '''
        if self.detector_type == 'mtcnn':
            detection_alignment_results = self.detector.detect_faces(person_img)
            for face in detection_alignment_results:
                return face
            return None
        elif self.detector_type == 'dlib_hog':
            face_bboxes = self.detector(person_img, 1)
            for face in face_bboxes:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                
                bbox_rectangle = face
                facial_landmark_position = self.landmark_extractor(person_img, bbox_rectangle)
                facial_landmark_position = face_utils.shape_to_np(facial_landmark_position)

                return ((x, y, w, h), facial_landmark_position)
            return None
        elif self.detector_type == 'cascade':
            face_bboxes = self.detector.detectMultiScale(person_img, scaleFactor=1.1, minNeighbors=5)
            for face in face_bboxes:
                x, y, w, h = face
                
                bbox_rectangle = dlib.rectangle(x, y, x+w, y+h)
                facial_landmark_position = self.landmark_extractor(person_img, bbox_rectangle)
                facial_landmark_position = face_utils.shape_to_np(facial_landmark_position)

                return ((x, y, w, h), facial_landmark_position)
            return None
        else:
            print('The detector not implemented')
