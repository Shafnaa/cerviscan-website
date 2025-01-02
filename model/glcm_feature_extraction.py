import cv2 as cv
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

import os
from tqdm import tqdm
import pandas as pd
import glob

def get_glcm_features(image_path):
    """
    Ekstraksi fitur dari matriks co-occurrence tingkat abu-abu (GLCM) untuk sebuah citra.

    Parameters:
        image_path (str): Jalur file ke citra yang akan dianalisis.

    Returns:
        list: Daftar nilai fitur yang diekstrak, meliputi:
              - contrast1 (float): Tingkat kontras dari GLCM.
              - correlation1 (float): Tingkat korelasi dari GLCM.
              - energy1 (float): Tingkat energi dari GLCM.
              - homogeneity1 (float): Tingkat homogenitas dari GLCM.
              - res_entropy (float): Entropi dari citra asli.
    """
    # Baca citra dari file
    image = cv.imread(image_path)

    # Konversi citra ke grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    # Parameter untuk GLCM
    distances = [1, 2]  # Jarak antar piksel
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]  # Sudut orientasi
    levels = 256  # Jumlah level intensitas abu-abu
    
    # Hitung GLCM
    glcm = graycomatrix(
        gray_image.astype(int),
        distances=distances,
        angles=angles,
        levels=levels,
        symmetric=True,
        normed=True
    )

    # Ekstraksi fitur dari GLCM
    contrast = graycoprops(glcm, prop='contrast')
    contrast1 = round(contrast.flatten()[0], 3)

    correlation = graycoprops(glcm, prop='correlation')
    correlation1 = round(correlation.flatten()[0], 3)

    energy = graycoprops(glcm, prop='energy')
    energy1 = round(energy.flatten()[0], 3)

    homogeneity = graycoprops(glcm, prop='homogeneity')
    homogeneity1 = round(homogeneity.flatten()[0], 3)
    
    # Hitung entropi dari citra asli
    res_entropy = round(entropy(image), 3)
    
    # Kembalikan nilai fitur
    return [contrast1, correlation1, energy1, homogeneity1, res_entropy]

def get_glcm_feature_names():
    """
    Mendapatkan nama fitur yang diekstrak dari matriks GLCM.

    Returns:
        list: Daftar nama fitur:
              - 'contrast': Tingkat kontras.
              - 'correlation': Tingkat korelasi.
              - 'energy': Tingkat energi.
              - 'homogeneity': Tingkat homogenitas.
              - 'entropy': Entropi citra.
    """
    return ['contrast', 'correlation', 'energy', 'homogeneity', 'entropy']
