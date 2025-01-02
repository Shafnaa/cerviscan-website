import numpy as np
import skimage
import cv2

from scipy.stats import skew

def get_lab_color_moment_features(image_path):
    """
    Extract color moment features from an image in the LAB color space.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        list: A list of mean, standard deviation, and skewness values for each channel (L, A, and B).
    """
    # Read the image
    image = cv2.imread(image_path)
    
    # Convert BGR to RGB color space
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Convert the RGB image to a numpy array
    image_array = np.array(rgb_image)
    
    # Normalize the RGB array
    rgb_img_normalized = [[[element / 255 for element in sublist] for sublist in inner_list] for inner_list in image_array]
    
    # Convert normalized RGB to LAB using skimage
    lab_image = skimage.color.rgb2lab(rgb_img_normalized)

    # Calculate mean, standard deviation, and skewness for each channel (L, A, and B)
    mean_l = np.mean(lab_image[:, :, 0])
    mean_a = np.mean(lab_image[:, :, 1])
    mean_b = np.mean(lab_image[:, :, 2])

    std_l = np.std(lab_image[:, :, 0])
    std_a = np.std(lab_image[:, :, 1])
    std_b = np.std(lab_image[:, :, 2])

    skew_l = skew(lab_image[:, :, 0].flatten())
    skew_a = skew(lab_image[:, :, 1].flatten())
    skew_b = skew(lab_image[:, :, 2].flatten())

    return [mean_l, mean_a, mean_b, std_l, std_a, std_b, skew_l, skew_a, skew_b]

def get_lab_color_moment_feature_names():
    """
    Get the names of the extracted features.

    Returns:
        list: A list of feature names.
    """
    return [
        'mean_l', 'mean_a', 'mean_b',
        'std_l', 'std_a', 'std_b',
        'skew_l', 'skew_a', 'skew_b'
    ]