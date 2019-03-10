import os
import scipy
import numpy as np
import cv2 as cv
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

"""
Logic:
If predict below some threshold: # new person
    trained_keyframe_id_list.append(new_person_keyframe)
    counter_id += 1
    keyframe_person_dict[new_person_keyframe] = counter_id
else:
    keyframe_person_dict[new_person_keyframe] = predicted 
"""

class FaceReidentification:
    def __init__(self, n_neighbors=3, pca_component=10):
        curr_dir = os.path.dirname(__file__)
        self.input_dir_path = os.path.join(curr_dir, '../keyframe_output/')
        
        self.pca = PCA(pca_component, whiten=True)
        self.knn_classifier = KNeighborsClassifier(n_neighbors=n_neighbors)
        self.svm_classifier = SVC(gamma='scale', probability=True)
        self.trained_keyframe_id_list = [0]
        self.keyframe_person_dict = {}
        self.keyframe_person_dict[0] = 0
        self.counter_person_id = 0
    
    def is_not_trained(self):
        return len(self.trained_keyframe_id_list) < 2

    def init_train_model(self, classifier='knn'):
        """
        Assumption: First train = 2 people with different id (floor and one person)
        """
        # Dataset Creation
        img_path_list = os.listdir(self.input_dir_path)
        label_list = []
        for img_path in img_path_list:
            keyframe_id = int(img_path.split('_')[0][1:])
            if keyframe_id == 0 or keyframe_id == 1:
                if keyframe_id not in self.trained_keyframe_id_list:
                    self.trained_keyframe_id_list.append(keyframe_id)
                    self.counter_person_id += 1
                    self.keyframe_person_dict[keyframe_id] = self.counter_person_id
                    label_list.append(keyframe_id)  # keyframe_id == label for the first person
                else:
                    label_list.append(self.keyframe_person_dict[keyframe_id])            
            
        img_path_list = [os.path.join(self.input_dir_path, img_path) for img_path in img_path_list]
        
        X_train_image_data = np.matrix(self.flatten_batch_image(self.read_batch_image(img_path_list)))
        self.pca.fit(X_train_image_data)
        
        X_train_pca = self.pca.transform(X_train_image_data)
        if classifier == 'svm':
            self.svm_classifier.fit(X_train_pca, label_list)
        else:
            self.knn_classifier.fit(X_train_pca, label_list)


        print('new trained keyframeid, keyframe dict')
        print(self.trained_keyframe_id_list)
        print(self.keyframe_person_dict)
        # Validate
        # self._validate(img_path_list, label_list)

    def update_model(self, new_keyframe_id, classifier='knn'):
        if new_keyframe_id not in self.trained_keyframe_id_list:
            self.trained_keyframe_id_list.append(new_keyframe_id)
            self.counter_person_id += 1
            self.keyframe_person_dict[new_keyframe_id] = self.counter_person_id

            self.train_model(classifier=classifier)
            print('new trained keyframeid, keyframe dict')
            print(self.trained_keyframe_id_list)
            print(self.keyframe_person_dict)
        
    def train_model(self, classifier='knn'):
        base_img_path_list = os.listdir(self.input_dir_path)
        custom_img_path_list = []
        label_list = []
        for img_path in base_img_path_list:
            keyframe_id = int(img_path.split('_')[0][1:])
            
            if keyframe_id in self.trained_keyframe_id_list:
                label_list.append(self.keyframe_person_dict[keyframe_id])

                custom_img_path_list.append(os.path.join(self.input_dir_path, img_path))
                
        X_train_image_data = np.matrix(self.flatten_batch_image(self.read_batch_image(custom_img_path_list)))
        self.pca.fit(X_train_image_data)
        
        X_train_pca = self.pca.transform(X_train_image_data)
        if classifier == 'svm':
            self.svm_classifier.fit(X_train_pca, label_list)
        else:
            self.knn_classifier.fit(X_train_pca, label_list)
            
    def predict_single_image(self, filename, classifier='knn'):
        file_path = os.path.join(self.input_dir_path, filename)
        X_test_image_data = np.matrix(self.read_image(file_path).flatten())
        X_test_pca = self.pca.transform(X_test_image_data)

        if classifier == 'svm':
            return self.svm_classifier.predict(X_test_pca)
        else:
            return self.knn_classifier.predict(X_test_pca)

    def predict_batch(self, keyframe_id, classifier='knn', return_proba=False):
        base_img_path_list = os.listdir(self.input_dir_path)
        custom_img_path_list = []
        for img_path in base_img_path_list:
            keyframe_id_iterator = int(img_path.split('_')[0][1:])
            if keyframe_id_iterator == keyframe_id:
                custom_img_path_list.append(os.path.join(self.input_dir_path, img_path))
        
        if len(custom_img_path_list) == 0: # if no keyframe found
            return None
        else:
            X_test_image_data = np.matrix(self.flatten_batch_image(self.read_batch_image(custom_img_path_list)))
            X_test_pca = self.pca.transform(X_test_image_data)

            if return_proba:
                if classifier == 'svm':
                    return self.svm_classifier.predict_proba(X_test_pca)
                else:
                    return self.knn_classifier.predict_proba(X_test_pca)
            else:
                if classifier == 'svm':
                    return self.svm_classifier.predict(X_test_pca)
                else:
                    return self.knn_classifier.predict(X_test_pca)

    def _validate(self, img_path_list, label_list, classifier='knn'):
            X_train_filename, X_test_filename, y_train_filename, y_test_filename = train_test_split(
                img_path_list, label_list, test_size=0.2, random_state=0
            )
            X_train_image_data = np.matrix(self.flatten_batch_image(self.read_batch_image(X_train_filename)))
            X_test_image_data = np.matrix(self.flatten_batch_image(self.read_batch_image(X_test_filename)))
            
            self.pca.fit(X_train_image_data)
            X_train_pca = self.pca.transform(X_train_image_data)
            X_test_pca = self.pca.transform(X_test_image_data)
            
            predicted = None
            if classifier == 'svm':
                self.svm_classifier.fit(X_train_pca, y_train_filename)
                predicted = self.svm_classifier.predict(X_test_pca)
            else:
                self.knn_classifier.fit(X_train_pca, y_train_filename)
                predicted = self.knn_classifier.predict(X_test_pca)

            print(accuracy_score(predicted, y_test_filename))
            print(f1_score(predicted, y_test_filename, average='weighted'))

    def read_batch_image(self, img_path_list, resized_dimension=(60,60)):
        image_data_list = []
        for img_path in img_path_list:
            img = cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), resized_dimension, interpolation=cv.INTER_AREA)
            image_data_list.append(img)
        return image_data_list
    
    def read_image(self, img_path, resized_dimension=(60, 60)):
        img = cv.resize(cv.imread(img_path, cv.IMREAD_GRAYSCALE), resized_dimension, interpolation=cv.INTER_AREA)
        return img

    def flatten_batch_image(self, image_data_list):
        flattened_image_list = []
        for i in range(len(image_data_list)):
            flattened_image = image_data_list[i].flatten()
            flattened_image_list.append(flattened_image)
        return flattened_image_list
    