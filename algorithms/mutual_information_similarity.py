import cv2
from sklearn.metrics import mutual_info_score

from algorithms.similarity import Similarity
import numpy as np


class MutualInformationSimilarity(Similarity):

    def __init__(self, path1, path2):
        super().__init__(path1, path2)


    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def get_score(self):
        # Calculate the mutual information
        hist_2d, _, _ = np.histogram2d(self.image1.ravel(), self.image2.ravel(), bins=20)
        mi = mutual_info_score(None, None, contingency=hist_2d)
        return mi
