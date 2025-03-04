# This is a parent class for other comparators
from abc import abstractmethod, ABC
import numpy as np

class Similarity(ABC):
    def __init__(self, path1: str, path2: str):
        self.image1 = self.preprocess_image(path1)
        self.image2 = self.preprocess_image(path2)

    @abstractmethod
    def preprocess_image(self, image_path):
        pass

    @abstractmethod
    def get_score(self):
        pass