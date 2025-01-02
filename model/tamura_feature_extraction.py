import cv2
import numpy as np
from model.lbp_feature_extraction import lbp_implementation

# Function to calculate Coarseness
def coarseness(image, kmax):
    """
    Calculate the coarseness feature of an image based on the Tamura texture features.

    Parameters:
        image (numpy.ndarray): Input grayscale image.
        kmax (int): Maximum size of the neighborhood window for averaging.

    Returns:
        float: The coarseness value of the image.
    """
    image = np.array(image)
    w = image.shape[0]
    h = image.shape[1]
    kmax = kmax if (np.power(2, kmax) < w) else int(np.log(w) / np.log(2))
    kmax = kmax if (np.power(2, kmax) < h) else int(np.log(h) / np.log(2))
    average_gray = np.zeros([kmax, w, h])
    horizon = np.zeros([kmax, w, h])
    vertical = np.zeros([kmax, w, h])
    Sbest = np.zeros([w, h])

    for k in range(kmax):
        window = np.power(2, k)
        for wi in range(w)[window:(w - window)]:
            for hi in range(h)[window:(h - window)]:
                average_gray[k][wi][hi] = np.sum(image[wi - window:wi + window, hi - window:hi + window])
        for wi in range(w)[window:(w - window - 1)]:
            for hi in range(h)[window:(h - window - 1)]:
                horizon[k][wi][hi] = average_gray[k][wi + window][hi] - average_gray[k][wi - window][hi]
                vertical[k][wi][hi] = average_gray[k][wi][hi + window] - average_gray[k][wi][hi - window]
        horizon[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))
        vertical[k] = horizon[k] * (1.0 / np.power(2, 2 * (k + 1)))

    for wi in range(w):
        for hi in range(h):
            h_max = np.max(horizon[:, wi, hi])
            h_max_index = np.argmax(horizon[:, wi, hi])
            v_max = np.max(vertical[:, wi, hi])
            v_max_index = np.argmax(vertical[:, wi, hi])
            index = h_max_index if (h_max > v_max) else v_max_index
            Sbest[wi][hi] = np.power(2, index)

    fcrs = np.mean(Sbest)
    return fcrs

# Function to calculate Contrast
def contrast(image):
    """
    Calculate the contrast feature of an image based on the Tamura texture features.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        float: The contrast value of the image.
    """
    image = np.array(image)
    image = np.reshape(image, (1, image.shape[0] * image.shape[1]))
    m4 = np.mean(np.power(image - np.mean(image), 4))
    v = np.var(image)
    std = np.power(v, 0.5)
    alfa4 = m4 / np.power(v, 2)
    fcon = std / np.power(alfa4, 0.25)
    return fcon

# Function to calculate Directionality
def directionality(image):
    """
    Calculate the directionality feature of an image based on the Tamura texture features.

    Parameters:
        image (numpy.ndarray): Input grayscale image.

    Returns:
        float: The directionality value of the image.
    """
    image = np.array(image, dtype='int64')
    h = image.shape[0]
    w = image.shape[1]
    convH = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    convV = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    deltaH = np.zeros([h, w])
    deltaV = np.zeros([h, w])
    theta = np.zeros([h, w])

    # Calculate deltaH
    for hi in range(h)[1:h - 1]:
        for wi in range(w)[1:w - 1]:
            deltaH[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convH))
    for wi in range(w)[1:w - 1]:
        deltaH[0][wi] = image[0][wi + 1] - image[0][wi]
        deltaH[h - 1][wi] = image[h - 1][wi + 1] - image[h - 1][wi]
    for hi in range(h):
        deltaH[hi][0] = image[hi][1] - image[hi][0]
        deltaH[hi][w - 1] = image[hi][w - 1] - image[hi][w - 2]

    # Calculate deltaV
    for hi in range(h)[1:h - 1]:
        for wi in range(w)[1:w - 1]:
            deltaV[hi][wi] = np.sum(np.multiply(image[hi - 1:hi + 2, wi - 1:wi + 2], convV))
    for wi in range(w):
        deltaV[0][wi] = image[1][wi] - image[0][wi]
        deltaV[h - 1][wi] = image[h - 1][wi] - image[h - 2][wi]
    for hi in range(h)[1:h - 1]:
        deltaV[hi][0] = image[hi + 1][0] - image[hi][0]
        deltaV[hi][w - 1] = image[hi + 1][w - 1] - image[hi][w - 1]

    deltaG = (np.absolute(deltaH) + np.absolute(deltaV)) / 2.0
    deltaG_vec = np.reshape(deltaG, (deltaG.shape[0] * deltaG.shape[1]))

    # Calculate theta
    for hi in range(h):
        for wi in range(w):
            if (deltaH[hi][wi] == 0 and deltaV[hi][wi] == 0):
                theta[hi][wi] = 0
            elif deltaH[hi][wi] == 0:
                theta[hi][wi] = np.pi
            else:
                theta[hi][wi] = np.arctan(deltaV[hi][wi] / deltaH[hi][wi]) + np.pi / 2.0
    theta_vec = np.reshape(theta, (theta.shape[0] * theta.shape[1]))

    n = 16
    t = 12
    cnt = 0
    hd = np.zeros(n)
    dlen = deltaG_vec.shape[0]
    for ni in range(n):
        for k in range(dlen):
            if ((deltaG_vec[k] >= t) and (theta_vec[k] >= (2 * ni - 1) * np.pi / (2 * n)) and (theta_vec[k] < (2 * ni + 1) * np.pi / (2 * n))):
                hd[ni] += 1
    hd = hd / np.mean(hd)
    hd_max_index = np.argmax(hd)
    fdir = 0
    for ni in range(n):
        fdir += np.power((ni - hd_max_index), 2) * hd[ni]
    return fdir

# Function to calculate Roughness
def roughness(fcrs, fcon):
    """
    Calculate the roughness feature as the sum of coarseness and contrast values.

    Parameters:
        fcrs (float): Coarseness value.
        fcon (float): Contrast value.

    Returns:
        float: The roughness value.
    """
    return fcrs + fcon

# Function to extract Tamura features
def get_tamura_features(image, lbp='off'):
    """
    Extract Tamura texture features from an image.

    Parameters:
        image (str): Path to the input image file.
        lbp (str): Option to apply LBP ('on' or 'off'). Default is 'off'.

    Returns:
        list: A list of Tamura texture features [Coarseness, Contrast, Directionality, Roughness].
    """
    if lbp == 'off':
        img = cv2.imread(image)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img = lbp_implementation(image)

    tamura_features = [
        coarseness(img, 5), 
        contrast(img), 
        directionality(img), 
        roughness(coarseness(img, 5), contrast(img))
    ]
    return tamura_features

# Function to extract Tamura features using LBP
def get_tamura_on(image):
    """
    Extract Tamura texture features from an image with LBP applied.

    Parameters:
        image (str): Path to the input image file.

    Returns:
        list: A list of Tamura texture features [Coarseness, Contrast, Directionality, Roughness].
    """
    return get_tamura_features(image, lbp='on')

# Function to get feature names
def get_tamura_feature_names():
    """
    Get the names of the Tamura texture features.

    Returns:
        list: A list of feature names ['Coarseness', 'Contrast', 'Directionality', 'Roughness'].
    """
    return ['Coarseness', 'Contrast', 'Directionality', 'Roughness']