import cv2 as cv
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.metrics.cluster import entropy

def get_glcm(image_path):
    image = cv.imread(image_path)

    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    distances = [1, 2]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    levels = 256
    
    glcm = graycomatrix(
        gray_image.astype(int),
        distances = distances,
        angles= angles,
        levels = levels,
        symmetric = True,
        normed=True
        )

    contrast = graycoprops(glcm, prop='contrast')
    contrast1 = contrast.flatten()[0]
    contrast1 = round(contrast1,3)

    correlation = graycoprops(glcm,prop='correlation')
    correlation1 = correlation.flatten()[0]
    correlation1 = round(correlation1,3)

    energy = graycoprops(glcm,prop='energy')
    energy1 = energy.flatten()[0]
    energy1 = round(energy1,3)

    homogeneity = graycoprops(glcm,prop='homogeneity') 
    homogeneity1 = homogeneity.flatten()[0]
    homogeneity1 = round(homogeneity1,3)
    
    res_entropy = entropy(image)
    res_entropy = round(res_entropy,3)
    
    return [contrast1, correlation1, energy1, homogeneity1, res_entropy]

def get_feature_name():
    return ['contrast', 'correlation', 'energy', 'homogeneity', 'entropy']

