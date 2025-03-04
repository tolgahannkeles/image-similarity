import os
from algorithms.ssim_similarity import SSIMSimilarity


class TestCases:

    def __init__(self):
        pass

    def test1(self):
        """
        This method tests the similarity of images under the test-images and test-images-horizontal.
        :return:
        """
        path_images = "images/test-images/"
        path_images_horizontal = "images/test-images-horizontal/"

        image_name_list = os.listdir(path_images)

        for image_name in image_name_list:
            image_name_horizontal = path_images_horizontal + image_name
            image_name = path_images + image_name

            print(SSIMSimilarity(image_name, image_name_horizontal).get_score())







if __name__ == "__main__":
    import tensorflow as tf

    print(tf.config.list_physical_devices())

    test = TestCases()
    test.test1()