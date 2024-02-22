# IMPORTS
from pathlib import Path
import os

import tensorflow.keras as tfk
import pandas as pd
import numpy as np

from train import ROOT_PATH, MODELS_FOLDER
from model import unet_model
from utils import make_blend, SaveCSV, separate_masks, encode_mask

# CONSTANTS/PARAMETERS
MODEL_NAME = 'model.01-val_loss 0.3522-dice 0.6840.keras'

DATASET_FOLDER = Path('E:\\airbus-ship-detection\\test_v2') #ROOT_PATH / 'test_v2'
BLENDED_OUTPUT_FOLDER = ROOT_PATH / 'output'

# LOADING THE DATASET
sample_set = pd.read_csv(ROOT_PATH / 'sample_submission_v2.csv')

#MASK FOR LIMITING THE DATASET
SAMPLES_LIMIT = 15

# INIT SAVER OBJECT
saver = SaveCSV() 

# CHECKING FOLDERS
if not BLENDED_OUTPUT_FOLDER.exists():
    BLENDED_OUTPUT_FOLDER.mkdir()


# ENTRY POINT
if __name__ == '__main__':
    images = os.listdir(DATASET_FOLDER) # Taking list of files in sample folder
    first_image_name = images[0]
    first_image = tfk.preprocessing.image.load_img(DATASET_FOLDER / first_image_name) # Loading first image
    image_width, image_height = first_image.size

    model = unet_model((image_width, image_height, 3),) # Init model
    model.load_weights(MODELS_FOLDER / MODEL_NAME)

    result_df = pd.DataFrame(columns=sample_set.columns) # Creating empty DataFrame for results
    count = 0

    # Taking each image
    for image in images:
        # Image loading and preprocessing
        image_name = image
        image = tfk.preprocessing.image.load_img(DATASET_FOLDER / image) # Loading image

        # Raise error if dataset shape is not homogenous
        if image.size != (image_width, image_height):
            raise ValueError(f'Image {image_name} has wrong size. Should be {image_width}x{image_height}. Sample dataset should be homogenous.')

        image_array: np.ndarray = tfk.preprocessing.image.img_to_array(image)/255 # Conveting the image to an NP array
        image_array = np.expand_dims(image_array, axis=0) # Add batch dimension

        # Predict
        mask: np.ndarray = model.predict(image_array) 

        # Blend mask and image
        blended = make_blend(image_name, mask, DATASET_FOLDER)
        blended.save(BLENDED_OUTPUT_FOLDER / (image_name + '.png'))

        # Divide mask if multiple objects
        masks = separate_masks(mask)

        if not masks:
                # If no objects found
                new_row = pd.DataFrame({'ImageId': image_name, 'EncodedPixels': np.nan}, index=[0])
                result_df = pd.concat([result_df, new_row], ignore_index = True)
        else:
            # If objects
            for item in masks:
                # Taking single mask
                item = encode_mask(item) # RLE encoding
                new_row = pd.DataFrame({'ImageId': image_name, 'EncodedPixels': item}, index=[0])
                result_df = pd.concat([result_df, new_row], ignore_index = True)

        # If limit is active
        if SAMPLES_LIMIT:
            count += 1
            if count == SAMPLES_LIMIT:
                break

    saver.save(result_df)



