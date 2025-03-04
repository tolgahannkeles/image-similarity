import os
import itertools
import pandas as pd
import torch
import torch.nn.functional as F
import cv2
import numpy as np
from algorithms.ssim_similarity import SSIMSimilarity
from algorithms.histogram_similarity import HistogramSimilarity
from algorithms.mutual_information_similarity import MutualInformationSimilarity
from tqdm import tqdm
import cv2
print(cv2.cuda.getCudaEnabledDeviceCount())  # Should return 1 or more if CUDA is enabled


def load_image_gpu(image_path):
    """ Resmi yükleyip GPU belleğine alır. """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Gri tonlamalı yükle
    img = cv2.resize(img, (256, 256))  # Boyutlandır (opsiyonel)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0  # Normalize et
    return img.to("cuda")


def ssim_gpu(img1, img2):
    """ SSIM hesaplamasını GPU üzerinde yapar. """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, 3, 1)
    mu2 = F.avg_pool2d(img2, 3, 1)

    sigma1_sq = F.avg_pool2d(img1 ** 2, 3, 1) - mu1 ** 2
    sigma2_sq = F.avg_pool2d(img2 ** 2, 3, 1) - mu2 ** 2
    sigma12 = F.avg_pool2d(img1 * img2, 3, 1) - mu1 * mu2

    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()


def histogram_similarity_gpu(img1_path, img2_path):
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    gpu_img1 = cv2.cuda_GpuMat()
    gpu_img2 = cv2.cuda_GpuMat()

    gpu_img1.upload(img1)
    gpu_img2.upload(img2)

    hist1 = cv2.cuda.calcHist([gpu_img1], [0], None, [256], [0, 256])
    hist2 = cv2.cuda.calcHist([gpu_img2], [0], None, [256], [0, 256])

    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def get_image_combinations(directory):
    files = [f for f in os.listdir(directory) if f.lower().endswith(('png', 'jpg', 'jpeg'))]
    return list(itertools.combinations(files, 2))


def analyze_and_write_to_excel(directory, excel_file):
    combinations = get_image_combinations(directory)
    analysis_results = []

    with pd.ExcelWriter(excel_file, mode='w') as writer:
        for img1, img2 in tqdm(combinations, desc="Processing combinations"):
            img1_gpu = load_image_gpu(os.path.join(directory, img1))
            img2_gpu = load_image_gpu(os.path.join(directory, img2))

            histogram_score = histogram_similarity_gpu(os.path.join(directory, img1), os.path.join(directory, img2))
            ssim_score = ssim_gpu(img1_gpu, img2_gpu)
            mutual_information_score = MutualInformationSimilarity(os.path.join(directory, img1),
                                                                   os.path.join(directory, img2)).get_score()

            analysis_results.append((img1, img2, ssim_score, histogram_score, mutual_information_score))

            # Write the current result to the Excel file
            df = pd.DataFrame(analysis_results,
                              columns=["Image 1", "Image 2", "ssim_score", "histogram_score",
                                       "mutual_information_score"])
            df.to_excel(writer, index=False)

    print(f"Results saved to {excel_file}.")


# Example usage
directory = "/mnt/d/kayseri_dijital_atlasi/dataset-copy/selected/train/images"  # Directory containing the images
file_name = "results.xlsx"
analyze_and_write_to_excel(directory, file_name)
