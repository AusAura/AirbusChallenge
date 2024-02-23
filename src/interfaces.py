# IMPORTS
from abc import ABC, abstractmethod
from pathlib import Path

import tensorflow.keras as tfk
import pandas as pd
import numpy as np


class DataGenerator(ABC, tfk.utils.Sequence):
    """
    ABSTRACT OF BASE DATA GENERATOR
    """

    def __init__(self, datapath: Path, batch_size: int, df_mask: pd.DataFrame):
        self.datapath = datapath  # Folder with the dataset
        self.batch_size = batch_size  # Amount of images in one batch
        self.df = df_mask.sample(
            frac=1
        )  # DataFrame mask that allows to limit our dataset artificially
        self.l = len(self.df) // batch_size  # Dataset length
        self.image_sizes = self.get_first_sizes(self)  # Get sizes of the first iamge

    def __len__(self):
        return self.l

    def on_epoch_end(self):
        pass

    @staticmethod
    def decode(mask_rle: np.ndarray, shape: tuple) -> np.ndarray:
        """
        Decodes RLE string
        mask_rle: run-length as string formated (start length)
        shape: (height,width) of the image

        Returns numpy array with 1 - mask, 0 - empty
        """
        img = np.zeros(
            int(shape[0] * shape[1]), dtype=np.float32
        )  # Create empty array for mask

        if not (type(mask_rle) is float):
            # If not NaN
            s = mask_rle.split()  # Separate each value
            # create 2 lists with odd and even values and save them as start positions and lenghts
            starts, lengths = [
                np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])
            ]
            starts -= 1
            ends = starts + lengths  # Finding end positions

            # Fill the masks with 1
            for lo, hi in zip(starts, ends):
                img[lo:hi] = 1.0

        return img.reshape((int(shape[0]), int(shape[1]))).T

    @staticmethod
    def get_first_sizes(self) -> tuple[int, int]:
        """
        Provides sizes of the first image in self.datapath

        Returns two integers (width, height)
        """
        first_image = tfk.preprocessing.image.load_img(
            self.datapath / self.df.iloc[0]["ImageId"]
        )  # Taking 1st image
        image_width, image_height = first_image.size  # Getting sizes

        return (image_width, image_height)

    @abstractmethod
    def __getitem__(self, index): ...


class CropGenerator(DataGenerator):
    """
    ABSTRACT OF DATA GENERATOR WITH CROP
    """

    @abstractmethod
    def crop(self): ...

    @abstractmethod
    def crop_mask(self): ...


class Save(ABC):
    """
    ABSTRACT OF SAVER FOR PREDICTED IMAGES
    """

    @abstractmethod
    def save(self, result_df: pd.DataFrame): ...
