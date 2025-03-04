from algorithms.similarity import Similarity
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
import  numpy as np
from scipy.spatial.distance import cosine

class FeatureSimilarity(Similarity):

    def __init__(self, path1: str, path2: str):
        super().__init__(path1, path2)

    def get_score(self):
        base_model = VGG16(weights='imagenet', include_top=False)
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)
        features_img1 = model.predict(self.image1)
        features_img2 = model.predict(self.image2)

        # Flatten features and calculate cosine similarity
        features_img1 = features_img1.flatten()
        features_img2 = features_img2.flatten()
        similarity = 1 - cosine(features_img1, features_img2)

        return similarity

    def preprocess_image(self, image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data