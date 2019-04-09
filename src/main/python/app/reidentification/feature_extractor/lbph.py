from facerec.feature import SpatialHistogram
import cv2 as cv

class LBPHFeatureExtractor:
    def __init__(self):
        self.extractor = SpatialHistogram()
        self.img_size = (60, 60)
        
    def extract(self, img_path):
        gray_img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
        gray_img = cv.resize(gray_img, self.img_size, interpolation=cv.INTER_AREA)
        return self.extractor.extract(gray_img)

    def extract_image(self, img):
        gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray_img = cv.resize(gray_img, self.img_size, interpolation=cv.INTER_AREA)
        return self.extractor.extract(gray_img)
        