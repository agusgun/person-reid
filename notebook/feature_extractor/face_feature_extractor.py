from .facenet_embedding import FacenetEmbedding
from .vgg_face_embedding import VGGFaceEmbedding
from .lbph import LBPHFeatureExtractor

class FaceFeatureExtractor:
    def __init__(self, pretrained_model_path, extractor_name='facenet'):
        if extractor_name == 'facenet':
            self.extractor = FacenetEmbedding(pretrained_model_path)
        elif extractor_name == 'vgg_face':
            self.extractor = VGGFaceEmbedding(pretrained_model_path, model_name='original')
        elif extractor_name == 'vgg_face_vgg16':
            self.extractor = VGGFaceEmbedding(pretrained_model_path, model_name='vgg16')
        elif extractor_name == 'vgg_face_resnet50':
            self.extractor = VGGFaceEmbedding(pretrained_model_path, model_name='resnet50')
        elif extractor_name == 'lbph':
            self.extractor = LBPHFeatureExtractor()
        else:
            print('Extractor not yet implemented')
            
    def extract(self, img_path):
        return self.extractor.extract(img_path)
    
    def extract_batch(self, img_path_list):
        features = []
        for img_path in img_path_list:
            features.append(self.extract(img_path))
        return features