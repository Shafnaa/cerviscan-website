import matplotlib.pyplot as plt
import numpy as np
from skimage import io, img_as_ubyte
from skimage.filters import threshold_multiotsu

def multiotsu_masking(image_path):
    image = io.imread(image_path)
    
    # Compute multi-Otsu thresholds
    threshold = threshold_multiotsu(image, classes=5)

    # Digitize (segment) the image based on the thresholds
    regions = np.digitize(image, bins=threshold)

    # Convert regions to uint8 explicitly to avoid the warning
    output = (regions * (255 // (regions.max() + 1))).astype(np.uint8)
    output[output < np.unique(output)[-1]] = 0
    output[output >= np.unique(output)[-1]] = 255

    return output