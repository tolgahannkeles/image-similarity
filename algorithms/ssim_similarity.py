from algorithms.similarity import Similarity
from skimage.metrics import structural_similarity as ssim
import cv2

class SSIMSimilarity(Similarity):

    def __init__(self, path1, path2):
        super().__init__(path1, path2)

    def preprocess_image(self, image_path):
        image = cv2.imread(image_path)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def get_score(self):
        # Compute SSIM between the two images
        score, _ = ssim(self.image1, self.image2, full=True)
        return score