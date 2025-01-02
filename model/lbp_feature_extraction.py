import numpy as np
import cv2

def get_pixel(img, center, x, y):
    """
    Get the binary value for a pixel based on its center value.

    Parameters:
        img (numpy.ndarray): The input grayscale image.
        center (int): The intensity value of the center pixel.
        x (int): The x-coordinate of the neighboring pixel.
        y (int): The y-coordinate of the neighboring pixel.

    Returns:
        int: 1 if the neighboring pixel's intensity is greater than or equal to the center, else 0.
    """
    new_value = 0
    try:
        if img[x][y] >= center:
            new_value = 1
    except IndexError:
        pass
    return new_value


def lbp_calculated_pixel(img, x, y):
    """
    Calculate the LBP value for the center pixel.

    Parameters:
        img (numpy.ndarray): The input grayscale image.
        x (int): The x-coordinate of the center pixel.
        y (int): The y-coordinate of the center pixel.

    Returns:
        int: The LBP value of the center pixel.
    """
    center = img[x][y]
    val_ar = []

    # Collect binary values from neighbors
    val_ar.append(get_pixel(img, center, x - 1, y - 1))  # top-left
    val_ar.append(get_pixel(img, center, x - 1, y))      # top
    val_ar.append(get_pixel(img, center, x - 1, y + 1))  # top-right
    val_ar.append(get_pixel(img, center, x, y + 1))      # right
    val_ar.append(get_pixel(img, center, x + 1, y + 1))  # bottom-right
    val_ar.append(get_pixel(img, center, x + 1, y))      # bottom
    val_ar.append(get_pixel(img, center, x + 1, y - 1))  # bottom-left
    val_ar.append(get_pixel(img, center, x, y - 1))      # left

    # Convert binary values to decimal
    power_val = [1, 2, 4, 8, 16, 32, 64, 128]
    val = sum(val_ar[i] * power_val[i] for i in range(len(val_ar)))

    return val


def lbp_implementation(path):
    """
    Generate the LBP image from the input image.

    Parameters:
        path (str): Path to the input image.

    Returns:
        numpy.ndarray: The resulting LBP image.
    """
    img_bgr = cv2.imread(path, 1)
    height, width, _ = img_bgr.shape

    # Convert the image to grayscale
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Initialize an empty image for LBP
    img_lbp = np.zeros((height, width), np.uint8)

    # Compute LBP for each pixel
    for i in range(height):
        for j in range(width):
            img_lbp[i, j] = lbp_calculated_pixel(img_gray, i, j)

    return img_lbp

def get_lbp_features(path):
    """
    Extract LBP features from the input image.

    Parameters:
        path (str): Path to the input image.

    Returns:
        list: A list containing mean, median, standard deviation, kurtosis, and skewness of the LBP image.
    """
    lbp_image = lbp_implementation(path).flatten()

    # Mean
    mean = np.mean(lbp_image)

    # Median
    median = np.median(lbp_image)

    # Standard Deviation
    std = np.std(lbp_image)
    n = len(lbp_image)

    # Kurtosis
    squared_differences = (lbp_image - mean) ** 4
    sum_of_squared_differences = np.sum(squared_differences)
    kurtosis = (4 * sum_of_squared_differences) / (n * std ** 4) - 3

    # Skewness
    skewness = (3 * (mean - median)) / std

    return [mean, median, std, kurtosis, skewness]


def get_lbp_feature_names():
    """
    Get the names of the extracted LBP features.

    Returns:
        list: A list of feature names.
    """
    return ['mean', 'median', 'std', 'kurtosis', 'skewness']