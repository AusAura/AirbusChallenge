# IMPORTS
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
from PIL import Image

from interfaces import Save


# CONSTANTS/PARAMETERS
BLEND_ALPHA = 0.4 # Brightness of mask on blended image


# CLASSES
class SaveCSV(Save):
    '''Save result in CSV format'''
    def save(self, result_df: pd.DataFrame) -> None:
        result_df.to_csv('result.csv', index=False)

# FUNCTIONS
def make_blend(image_id: str, mask: np.ndarray, sample_folder: Path) -> None:
    '''
    Blends mask and original image
    :image_id: str - image name
    :mask: np.ndarray - multiple mask

    Returns None
    '''
    # Upload original image and convert
    original_image = Image.open(sample_folder / image_id)
    original_image = original_image.convert('RGBA')

    # Create mask array
    mask = mask.reshape(mask.shape[1], mask.shape[2])
    mask_image = Image.fromarray((mask * 255).astype(np.uint8)).convert('RGBA')

    # Blend
    result_image = Image.blend(original_image, mask_image, alpha=BLEND_ALPHA) 
    return result_image

def encode_mask(mask: np.ndarray) -> np.ndarray:
    '''
    Encodes current single mask with RLE string
    :mask: np.ndarray

    Returns RLE string in np.array
    '''
    RLE_string = ''
    prev_pixel = 0
    run_length = 0

    for row in mask:
        for pixel in row:            
            run_length += 1 # Count 1 pixel each time

            if prev_pixel != pixel:
                # If pixel changes
                RLE_string =  RLE_string + str(run_length) + ' '
                run_length = 0
                prev_pixel = pixel

    RLE_string = RLE_string.rstrip() # Deleting space from the right
    return np.array(RLE_string)

def separate_masks(mask: np.ndarray) -> list[np.ndarray]:
    '''
    Divides mask with multiple objects to the list of single ones
    :mask: np.ndarray - mask with multiple objects

    Returns list of masks
    '''
    mask = np.squeeze(mask)  # Removing a dimension containing only one value
    mask = (mask * 255).astype(np.uint8) # Standartize

    # Thresholding of a mask
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

    # Finding contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    individual_masks = []

    for contour in contours:
        # For each contour
        mask = np.zeros_like(mask) # Create empty array with the same shape
        cv2.drawContours(mask, [contour], 0, 255, -1) # Fill the contour
        individual_masks.append(mask) # Add mask

    return individual_masks