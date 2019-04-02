import keras
from keras.models import Model, Sequential
from keras.layers import Input, Convolution2D, ZeroPadding2D, MaxPooling2D, Flatten, Dense, Dropout, Activation
from keras.preprocessing.image import load_img, save_img, img_to_array
from keras.applications.imagenet_utils import preprocess_input
import numpy as np
from keras_vggface.vggface import VGGFace
from keras_vggface import utils
import cv2 as cv

class VGGFaceEmbedding:
    def __init__(self, pretrained_model_path, model_name='original'):
        if model_name == 'original':
            model = Sequential()
            model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
            model.add(Convolution2D(64, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(64, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))

            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(128, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))

            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(256, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))

            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))

            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(ZeroPadding2D((1,1)))
            model.add(Convolution2D(512, (3, 3), activation='relu'))
            model.add(MaxPooling2D((2,2), strides=(2,2)))

            model.add(Convolution2D(4096, (7, 7), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Convolution2D(4096, (1, 1), activation='relu'))
            model.add(Dropout(0.5))
            model.add(Convolution2D(2622, (1, 1)))
            model.add(Flatten())
            model.add(Activation('softmax'))
            
            model = model
            model.load_weights(pretrained_model_path)
            self.descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
            self.preprocess_version = None
        elif model_name == 'vgg16':
            self.descriptor = VGGFace(model='vgg16', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            self.preprocess_version = 1
        elif model_name == 'resnet50':
            self.descriptor = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
            self.preprocess_version = 2
        else:
            print('Model not yet implemented')
    
    def preprocess_image(self, img_path):
        img = load_img(img_path, target_size=(224, 224))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        if self.preprocess_version:
            img = utils.preprocess_input(img, version=self.preprocess_version)
        else:
            img = preprocess_input(img)
            
        return img
    
    def extract(self, img_path):
        return self.descriptor.predict(self.preprocess_image(img_path))[0]

    def extract_image(self, img):
        img = cv.resize(img, (224, 224), interpolation=cv.INTER_AREA).astype(np.float32)
        img = np.expand_dims(img, axis=0)
        if self.preprocess_version:
            img = utils.preprocess_input(img, version=self.preprocess_version)
        else:
            img = preprocess_input(img)
            
        return self.descriptor.predict(img)[0]