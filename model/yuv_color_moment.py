import numpy as np
from scipy.stats import skew

from PIL import Image
import numpy as np

def get_yuv_color_moment_features(image_path):
    image = Image.open(image_path)
    
    # Convert image to numpy array
    image_array = np.array(image)
    
    # YUV conversion matrix
    yuv_matrix = np.array([[0.299, 0.587 , 0.114],
                        [-0.147 , -0.289 , 0.436],
                        [0.615 , -0.515 , 0.100]])

    # Shape of the Image
    image_shape = image_array.shape

    # Create an array of zeros with the same shape
    yuv_image = np.zeros(image_shape, dtype=np.float64)

    # Iterate over the RGB array
    for i in range(image_shape[0]):
        for j in range(image_shape[1]):
            yuv_image[i][j] = np.dot(yuv_matrix, image_array[i][j])

    # Mean
    mean_y = np.mean(yuv_image[:, :, 0])
    mean_u = np.mean(yuv_image[:, :, 1])
    mean_v = np.mean(yuv_image[:, :, 2])
            
    # Std
    std_y = np.std(yuv_image[:, :, 0])
    std_u = np.std(yuv_image[:, :, 1])
    std_v = np.std(yuv_image[:, :, 2])

    # Skewness
    skew_y = skew(yuv_image[:, :, 0].flatten())
    skew_u = skew(yuv_image[:, :, 1].flatten())
    skew_v = skew(yuv_image[:, :, 2].flatten())
    
    return [mean_y, mean_u, mean_v, std_y, std_u, std_v, skew_y, skew_u, skew_v]

def get_feature_name():
    return ['mean_y', 'mean_u', 'mean_v', 'std_y', 'std_u', 'std_v', 'skew_y', 'skew_u', 'skew_v',]