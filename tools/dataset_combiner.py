import os
import uuid
import random
import string
import shutil

train_path = "/mnt/d/kayseri_dijital_atlasi/dataset-copy/selected/train"
test_path = "/mnt/d/kayseri_dijital_atlasi/dataset-copy/selected/test"

def generate_random_id(length=6):
    characters = string.ascii_letters + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

def combine_images():
    images_path = os.path.join(train_path, "images")
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    for category in os.listdir(train_path):
        if category == "images":
            continue
        category_path = os.path.join(train_path, category)
        if os.path.isdir(category_path):
            for image in os.listdir(category_path):
                image_path = os.path.join(category_path, image)
                new_image_path = os.path.join(images_path, category + "-" + generate_random_id() + ".jpg")
                shutil.copy(image_path, new_image_path)

combine_images()