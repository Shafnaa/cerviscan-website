import cv2

def get_segmented_image(original_image, mask_path):
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the mask image has the same dimensions as the original image
    mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]))

    # Convert the grayscale mask to a 3-channel image
    mask_image_3channel = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2RGB)

    # Perform bitwise AND operation
    dest_and = cv2.bitwise_and(original_image, mask_image_3channel)

    return dest_and
