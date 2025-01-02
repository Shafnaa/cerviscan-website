import numpy as np

from scipy.stats import skew
from PIL import Image

def get_yuv_color_moment_features(image_path):
    """
    Extract color moment features from an image in the YUV color space.

    Parameters:
        image_path (str): Path to the image file.

    Returns:
        list: A list of mean, standard deviation, and skewness values for each channel (Y, U, and V).
    """
    # Read the image
    image = Image.open(image_path)
    
    # Convert the image to a numpy array
    image_array = np.array(image)
    
    # RGB to YUV conversion matrix
    yuv_matrix = np.array([
        [0.299, 0.587, 0.114],
        [-0.147, -0.289, 0.436],
        [0.615, -0.515, 0.100]
    ])

    # Get the dimensions of the image
    image_shape = image_array.shape

    # Prepare an empty array to store YUV values
    yuv_image = np.zeros(image_shape, dtype=np.float64)

    # Perform RGB to YUV color space conversion
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            yuv_image[i, j] = np.dot(yuv_matrix, image_array[i, j])

    # Calculate mean, standard deviation, and skewness for each channel (Y, U, and V)
    mean_y = np.mean(yuv_image[:, :, 0])
    mean_u = np.mean(yuv_image[:, :, 1])
    mean_v = np.mean(yuv_image[:, :, 2])

    std_y = np.std(yuv_image[:, :, 0])
    std_u = np.std(yuv_image[:, :, 1])
    std_v = np.std(yuv_image[:, :, 2])

    skew_y = skew(yuv_image[:, :, 0].flatten())
    skew_u = skew(yuv_image[:, :, 1].flatten())
    skew_v = skew(yuv_image[:, :, 2].flatten())

    return [mean_y, mean_u, mean_v, std_y, std_u, std_v, skew_y, skew_u, skew_v]

def get_yuv_color_moment_feature_names():
    """
    Get the names of the extracted features.

    Returns:
        list: A list of feature names.
    """
    return [
        'mean_y', 'mean_u', 'mean_v',
        'std_y', 'std_u', 'std_v',
        'skew_y', 'skew_u', 'skew_v'
    ]
