# IMPORTS
from pathlib import Path

import pandas as pd
import numpy as np
import tensorflow.keras as tfk
import tensorflow.keras.backend as K

from interfaces import CropGenerator


# CROP DATA GENERATOR CLASSES
class Crop2x2Generator(CropGenerator):
    '''This data generator crops the images for the model to 2x2 pieces'''
    def __init__(self, datapath: Path, batch_size: int, df_mask: pd.DataFrame):
        super().__init__(datapath, batch_size, df_mask)
        self.part_size = int(self.image_sizes[0]/2)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # Creating empty NP arrays for image and mask
        mask = np.empty((self.batch_size, int(self.image_sizes[0]/2), int(self.image_sizes[1]/2)), np.float32)
        image = np.empty((self.batch_size, int(self.image_sizes[0]/2), int(self.image_sizes[1]/2), 3), np.float32)

        for item in range(self.batch_size):
            # Single batch processing
            image_file = tfk.preprocessing.image.load_img(self.datapath / self.df.iloc[index*self.batch_size+item]['ImageId']) # Uploading the image from the mask DataFrame
            image_array = tfk.preprocessing.image.img_to_array(image_file)/255 # Conveting the image to an NP array
            image_width, image_height = image_file.size

            # Raise error if dataset shape is not homogenous
            if self.image_sizes != (image_width, image_height):
                raise ValueError(f'Image {item} has wrong size. Should be {image_width}x{image_height}. Sample dataset should be homogenous.')

            # Retrieving, decoding the mask of the image and cropping part with maximum ship's area
            mask[item], i = self.crop_mask(
                self.decode(
                    self.df.iloc[index*self.batch_size+item]['EncodedPixels'], 
                    (image_width, image_height)
                )
            )
            image[item] = self.crop(image_array, i) # Croping and retrieving part with maximum ship's area
        return image, mask
    
    def crop(self, img: np.ndarray, i: int) -> np.ndarray:
        """
        :img: np.ndarray - original image
        :i: int 0-8 - image index from crop: 0 1
                                             2 3

        returns: np.ndarray with cropped image
        """
        return img[(i//2)*self.part_size:((i//2)+1)*self.part_size, (i%2)*self.part_size:(i%2+1)*self.part_size]

    def crop_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Returns crop image, crop index with maximum ships area
        :img: np.ndarray - original mask

        returns: np.ndarray with cropped mask
        """
        i = K.argmax((
            K.sum(self.crop(img, 0)),
            K.sum(self.crop(img, 1)),
            K.sum(self.crop(img, 2)),
            K.sum(self.crop(img, 3)),
        ))
        return (self.crop(img, i), i) # Taking the found part
    

class Crop3x3Generator(CropGenerator):
    '''This data generator crops the images for the model to 3x3 pieces'''
    def __init__(self, datapath: Path, batch_size: int, df_mask: pd.DataFrame):
        super().__init__(datapath, batch_size, df_mask)
        self.part_size = int(self.image_sizes[0]/3)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # Creating empty NP arrays for image and mask
        mask = np.empty((self.batch_size, int(self.image_sizes[0]/3) , int(self.image_sizes[1]/3)), np.float32)
        image = np.empty((self.batch_size, int(self.image_sizes[0]/3), int(self.image_sizes[1]/3), 3), np.float32)

        for item in range(self.batch_size):
            # Single batch processing
            image_file = tfk.preprocessing.image.load_img(self.datapath / self.df.iloc[index*self.batch_size+item]['ImageId']) # Uploading the image from the mask DataFrame
            image_array = tfk.preprocessing.image.img_to_array(image_file)/255 # Conveting the image to an NP array
            image_width, image_height = image_file.size

            # Raise error if dataset shape is not homogenous
            if self.image_sizes != (image_width, image_height):
                raise ValueError(f'Image {item} has wrong size. Should be {image_width}x{image_height}. Sample dataset should be homogenous.')
            
            # Retrieving, decoding the mask of the image and cropping part with maximum ship's area
            mask[item], i = self.crop_mask(
                self.decode(
                    self.df.iloc[index*self.batch_size+item]['EncodedPixels'], 
                    (image_width, image_height)
                )
            )
            image[item] = self.crop(image_array, i) # Croping and retrieving part with maximum ship's area
        return image, mask

    def crop(self, img: np.ndarray, i: int) -> np.ndarray:
        """
        :img: np.ndarray - original image
        :i: int 0-8 - image index from crop: 0 1 2
                                             3 4 5
                                             6 7 8
        returns: np.ndarray with cropped image
        """
        return img[(i//3)*self.part_size:((i//3)+1)*self.part_size, (i%3)*self.part_size:(i%3+1)*self.part_size] # Taking the required part of the original image

    def crop_mask(self, img: np.ndarray) -> np.ndarray:
        """
        Returns crop image, crop index with maximum ships area
        :img: np.ndarray - original mask

        returns: np.ndarray with cropped mask
        """
        # Idetifying the part with the most of the ship area
        i = K.argmax((
            K.sum(self.crop(img, 0)),
            K.sum(self.crop(img, 1)),
            K.sum(self.crop(img, 2)),
            K.sum(self.crop(img, 3)),
            K.sum(self.crop(img, 4)),
            K.sum(self.crop(img, 5)),
            K.sum(self.crop(img, 6)),
            K.sum(self.crop(img, 7)),
            K.sum(self.crop(img, 8)),
        ))
        return (self.crop(img, i), i) # Taking the found part


class NoCropGenerator(CropGenerator):
    '''This data generator DOES NOT crops the original images for the model'''
    def __init__(self, datapath: Path, batch_size: int, df_mask: pd.DataFrame):
        super().__init__(datapath, batch_size, df_mask)
        self.part_size = self.image_sizes[0]

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray]:
        # Creating empty NP arrays for image and mask
        mask = np.empty((self.batch_size, self.image_sizes[0] , self.image_sizes[1]), np.float32)
        image = np.empty((self.batch_size, self.image_sizes[0], self.image_sizes[1], 3), np.float32)

        for item in range(self.batch_size):
            # Single batch processing
            image_file = tfk.preprocessing.image.load_img(self.datapath / self.df.iloc[item]['ImageId']) # Uploading the image from the mask DataFrame
            image_array = tfk.preprocessing.image.img_to_array(image_file)/255 # Conveting the image to an NP array
            image_width, image_height = image_file.size

            # Raise error if dataset shape is not homogenous
            if self.image_sizes != (image_width, image_height):
                raise ValueError(f'Image {item} has wrong size. Should be {image_width}x{image_height}. Sample dataset should be homogenous.')

            mask[item] = self.decode(
                    self.df.iloc[index*self.batch_size+item]['EncodedPixels'], # Retrieving, decoding the mask of the image and cropping part with maximum ship's area
                    (image_width, image_height)
                )
            image[item] = image_array
        return image, mask


data_generator = Crop3x3Generator # Init the selected data generator