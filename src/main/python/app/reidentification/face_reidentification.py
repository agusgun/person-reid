import os
import scipy
import numpy as np
import cv2 as cv
from .feature_extractor.face_feature_extractor import FaceFeatureExtractor
import tensorflow as tf
from keras import backend as K
from scipy.spatial.distance import euclidean

class FaceReidentification:
    def __init__(self, extractor_name='vgg_face_resnet'):
        curr_dir = os.path.dirname(__file__)
        FACENET_MODEL_PATH = os.path.join(curr_dir, '../model_data/facenet/20180402-114759')
        VGGFACE_MODEL_PATH = os.path.join(curr_dir, '../model_data/vgg_face_weights.h5')
        if extractor_name == 'facenet':
            print('Load Facenet')
            self.feature_extractor = FaceFeatureExtractor(FACENET_MODEL_PATH, extractor_name='facenet')
        elif extractor_name == 'vgg_face':
            print('Load VGGFace')
            self.feature_extractor = FaceFeatureExtractor(VGGFACE_MODEL_PATH, extractor_name='vgg_face')
        elif extractor_name == 'vgg_face_resnet':
            print('Load VGGFace RESNET')
            self.feature_extractor = FaceFeatureExtractor(None, extractor_name='vgg_face_resnet50')
        elif extractor_name == 'vgg_face_vgg16':
            print('Load VGGFace VGG16')
            self.feature_extractor = FaceFeatureExtractor(None, extractor_name='vgg_face_vgg16')
        elif extractor_name == 'lbph':
            print('Use LBPH')
            self.feature_extractor = FaceFeatureExtractor(None, extractor_name='lbph')
        else:
            print('Feature extractor not implemented')

    def extract_feature_from_keyframes(self, input_dir_path, keyframe_id,):
        keyframe_dir_path = os.path.join(input_dir_path, str(keyframe_id))
        keyframe_file_paths = os.listdir(keyframe_dir_path)
        keyframe_file_paths = [os.path.join(keyframe_dir_path, path) for path in keyframe_file_paths]
        features = self.feature_extractor.extract_batch(keyframe_file_paths)
        return features

    def predict(self, image_representation_database, image_representation_label, feature, min_distance):
        minimum_label = None
        minimum_distance = min_distance

        for idx, image_representations in enumerate(image_representation_database):
            for image_representation in image_representations:
                distance = euclidean(image_representation, feature)
                if distance < minimum_distance:
                    minimum_distance = distance
                    minimum_label = image_representation_label[idx]
        print(minimum_label, minimum_distance)
        return minimum_label
    
    def predict_and_find_match(self, image_representation_database, image_paths, image_representation_label, feature, min_distance):
        minimum_label = None
        minimum_distance = min_distance
        minimum_path = os.path.join(os.path.dirname(__file__), '../assets/default_none.png')

        for idx, image_representations in enumerate(image_representation_database):
            for idx2, image_representation in enumerate(image_representations):
                distance = euclidean(image_representation, feature)
                if distance < minimum_distance:
                    minimum_distance = distance
                    minimum_label = image_representation_label[idx]
                    minimum_path = image_paths[idx][idx2]
        print('PNFM', minimum_label, minimum_distance, minimum_path)
        return (minimum_label, minimum_path)

    def predict_batch(self, image_representation_database, image_representation_label, features, min_distance):
        predictions = []
        for feature in features:
            predictions.append(self.predict(image_representation_database, image_representation_label, feature, min_distance))
        return predictions

    def predict_and_find_match_batch(self, image_representation_database, image_paths, image_representation_label, features, min_distance):
        predictions = []
        for feature in features:
            predictions.append(self.predict_and_find_match(image_representation_database, image_paths, image_representation_label, feature, min_distance))
        return predictions