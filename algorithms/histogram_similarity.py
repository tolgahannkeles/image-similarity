import cv2

from algorithms.similarity import Similarity


class HistogramSimilarity(Similarity):

    def __init__(self, path1, path2):
        super().__init__(path1, path2)

    def preprocess_image(self, image_path):
        return cv2.imread(image_path)

    def get_score(self):

        hist1 = cv2.calcHist([self.image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([self.image2], [0], None, [256], [0, 256])

        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()

        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

