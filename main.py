# Import necessary libraries
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from scipy.spatial.distance import cosine
from sklearn.metrics import mutual_info_score


# Function to calculate SSIM
def calculate_ssim(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)

    # Compute SSIM between the two images
    score, _ = ssim(grayA, grayB, full=True)
    return score


# Function to extract features using VGG16 and calculate cosine similarity
def calculate_feature_similarity(image_path1, image_path2):
    # Load pre-trained VGG16 model
    base_model = VGG16(weights='imagenet', include_top=False)
    model = Model(inputs=base_model.input, outputs=base_model.get_layer('block5_pool').output)

    # Function to preprocess image for VGG16
    def preprocess_image(image_path):
        img = image.load_img(image_path, target_size=(224, 224))
        img_data = image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        return img_data

    # Preprocess the two images
    img1 = preprocess_image(image_path1)
    img2 = preprocess_image(image_path2)

    # Extract features
    features_img1 = model.predict(img1)
    features_img2 = model.predict(img2)

    # Flatten features and calculate cosine similarity
    features_img1 = features_img1.flatten()
    features_img2 = features_img2.flatten()
    similarity = 1 - cosine(features_img1, features_img2)

    return similarity

def calculate_histogram_similarity(imageA, imageB, method=cv2.HISTCMP_CORREL):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # Calculate histograms
    histA = cv2.calcHist([grayA], [0], None, [256], [0, 256])
    histB = cv2.calcHist([grayB], [0], None, [256], [0, 256])
    
    # Normalize histograms
    histA = cv2.normalize(histA, histA).flatten()
    histB = cv2.normalize(histB, histB).flatten()
    
    # Compute the histogram similarity
    similarity = cv2.compareHist(histA, histB, method)
    return similarity

def calculate_orb_similarity(imageA, imageB):
    # Initialize ORB detector
    orb = cv2.ORB_create()
    
    # Detect keypoints and descriptors
    keypointsA, descriptorsA = orb.detectAndCompute(imageA, None)
    keypointsB, descriptorsB = orb.detectAndCompute(imageB, None)
    
    # Initialize BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(descriptorsA, descriptorsB)
    
    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)
    
    # Calculate similarity score
    similarity = len(matches) / max(len(keypointsA), len(keypointsB))
    return similarity

# Function to calculate mutual information similarity
def calculate_mutual_information(imageA, imageB):
    # Convert images to grayscale
    grayA = cv2.cvtColor(imageA, cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(imageB, cv2.COLOR_BGR2GRAY)
    
    # Calculate the mutual information
    hist_2d, _, _ = np.histogram2d(grayA.ravel(), grayB.ravel(), bins=20)
    mi = mutual_info_score(None, None, contingency=hist_2d)
    return mi

# Load sample images
path_a="images/128/1.jpg"
path_b="images/128/1.jpg"

imageA = cv2.imread(path_a)
imageB = cv2.imread(path_b)

# Calculate SSIM
ssim_score = calculate_ssim(imageA, imageB)

# Calculate feature-based similarity using VGG16

feature_similarity = calculate_feature_similarity(path_a, path_b)

# Calculate histogram similarity
histogram_similarity = calculate_histogram_similarity(imageA, imageB)

# Calculate ORB similarity
orb_similarity = calculate_orb_similarity(imageA, imageB)

# Calculate mutual information similarity
mutual_information_similarity = calculate_mutual_information(imageA, imageB)

print(f"SSIM score: {ssim_score}")
print(f"Feature-based similarity (Cosine similarity): {feature_similarity}")
print(f"Histogram similarity: {histogram_similarity}")
print(f"ORB similarity: {orb_similarity}")
print(f"Mutual Information similarity: {mutual_information_similarity}")
