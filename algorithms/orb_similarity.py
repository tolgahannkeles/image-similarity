# orb_similarity.py

import cv2

class OrbSimilarity:
    def __init__(self, image1_path, image2_path):
        self.image1_path = image1_path
        self.image2_path = image2_path

    def get_score(self):
        image1 = cv2.imread(self.image1_path, cv2.IMREAD_GRAYSCALE)
        image2 = cv2.imread(self.image2_path, cv2.IMREAD_GRAYSCALE)

        orb = cv2.ORB_create()
        keypointsA, descriptorsA = orb.detectAndCompute(image1, None)
        keypointsB, descriptorsB = orb.detectAndCompute(image2, None)

        if len(keypointsA) == 0 or len(keypointsB) == 0:
            return 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptorsA, descriptorsB)
        matches = sorted(matches, key=lambda x: x.distance)

        similarity = len(matches) / max(len(keypointsA), len(keypointsB))
        return similarity