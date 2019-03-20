import tensorflow as tf
from facenet import facenet
import cv2 as cv


class FacenetEmbedding:
    def __init__(self, model_dir_path):
        self.img_size = (160, 160)
        with tf.Graph().as_default():
            self.sess = tf.InteractiveSession()
            facenet.load_model(model_dir_path)
            self.images_placeholder = tf.get_default_graph().get_tensor_by_name('input:0')
            self.images_placeholder = tf.image.resize_images(self.images_placeholder, self.img_size)
            self.embeddings_placeholder = tf.get_default_graph().get_tensor_by_name('embeddings:0')
            self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name('phase_train:0')
        
    def extract(self, img_path):
        img = cv.imread(img_path, 1) # RGB image
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        resized_img = cv.resize(img, self.img_size, interpolation=cv.INTER_AREA)
        prewithened = facenet.prewhiten(resized_img)

        # Get Embedding Here
        reshaped_img = prewithened.reshape(-1, self.img_size[0], self.img_size[1], 3)
        feed_dict = {self.images_placeholder:reshaped_img, self.phase_train_placeholder:False}
        features = self.sess.run(self.embeddings_placeholder, feed_dict=feed_dict)
        return features[0]
    
    def _image_to_embedding_batch(self, img_path_list):
        images = []
        for img_path in img_path_list:
            img = cv.imread(img_path, 1) # RGB image
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            resized_img = cv.resize(img, self.img_size, interpolation=cv.INTER_AREA)
            prewithened = facenet.prewhiten(resized_img)
            images.append(prewithened)
            
        images = np.array(images)
        reshaped_images = images.reshape(-1, self.img_size[0], self.img_size[1], 3)
        feed_dict = {self.images_placeholder:reshaped_images, self.phase_train_placeholder:False}
        embedding_result = self.sess.run(self.embeddings_placeholder, feed_dict=feed_dict)
        return embedding_result
        
    def close_session(self):
        self.sess.close()
        tf.reset_default_graph()