import numpy as np

from scipy.stats import skew
from PIL import Image

def get_rgb_color_moment_features(image_path):
    """
    Extract color moment features from an image in the RGB color space.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        list: A list of mean, standard deviation, and skewness values for each channel (R, G, and B).

    Raises:
        ValueError: If the image is not in RGB format.
    """
    # Open the image
    image = Image.open(image_path)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # Ensure the image has three color channels (in case of grayscale)
    if len(image_array.shape) < 3 or image_array.shape[2] != 3:
        raise ValueError(f"Image at {image_path} is not in RGB format.")

    # Calculate mean, standard deviation, and skewness for each channel (R, G, and B)
    mean_r = np.mean(image_array[:, :, 0])
    mean_g = np.mean(image_array[:, :, 1])
    mean_b = np.mean(image_array[:, :, 2])

    std_r = np.std(image_array[:, :, 0])
    std_g = np.std(image_array[:, :, 1])
    std_b = np.std(image_array[:, :, 2])

    skew_r = skew(image_array[:, :, 0].flatten())
    skew_g = skew(image_array[:, :, 1].flatten())
    skew_b = skew(image_array[:, :, 2].flatten())

    return [mean_r, mean_g, mean_b, std_r, std_g, std_b, skew_r, skew_g, skew_b]

def get_rgb_color_moment_feature_names():
    """
    Get the names of the extracted features.

    Returns:
        list: A list of feature names.
    """
    return [
        'mean_r', 'mean_g', 'mean_b',
        'std_r', 'std_g', 'std_b',
        'skew_r', 'skew_g', 'skew_b'
    ]