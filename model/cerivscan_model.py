from model.rgb_to_gray import rgb_to_gray_converter
from model.multiotsu_segmentation import multiotsu_masking
from model.bitwise_operation import get_segmented_image

from cerviscan_feature_extraction import get_cerviscan_features

import os
import pickle
import matplotlib.pyplot as plt 


def cerviscanModel(image_path, image_output):
    image = plt.imread(image_path)
    
    # Convert RBG Image to Grayscale Image
    gray_image = rgb_to_gray_converter(image_path)
    output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_gray.jpg")
    plt.imsave(output_path, gray_image, cmap="gray")
    
    # Get Image Masking
    mask_image = multiotsu_masking(gray_image)
    output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_mask.jpg")
    plt.imsave(output_path, mask_image, cmap="gray")
    
    # Segment Image Using Image Masking
    segmented_image = get_segmented_image(image, mask_image)
    
    output_path = os.path.join(f'{image_output}', f"{os.path.basename(image_path)[:-4]}_segmented.jpg")
    plt.imsave(output_path, segmented_image, cmap="gray")
    
    features = get_cerviscan_features(f'./{image_output}/{os.path.basename(image_path)[:-4]}_segmented.jpg')
    
    model = pickle.load(open('./xgb_best', 'rb'))
    result = model.predict(features)
    
    return features, result
    
    
    