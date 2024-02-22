# IMPORTS
import pickle
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from PIL import ImageFile
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split

from model import unet_model
from crop import data_generator

ImageFile.LOAD_TRUNCATED_IMAGES = True ## to avoid OSError during fit


# CONSTANTS/PARAMETERS
ROOT_PATH = Path(__file__).resolve().parent.parent

MODEL_NAME = 'last_model_weights.keras'
HISTORY_NAME = 'last_model_history.pickle'
MODELS_FOLDER = ROOT_PATH / 'models'

TRAIN_FOLDER = Path('E:\\airbus-ship-detection\\train_v2') # ROOT_PATH / 'train_v2'
TRAIN_DATASET_FILE_FOLDER = ROOT_PATH

EMPTY_IMAGES = 150 #000 # how many empty images will take from the train set
SHIPS_IMAGES = 81 #000 # how many images with ships/masks will take from the train set

EPOCHS = 2
ADAM_RATE = 0.001
BATCH_SIZE = 16
TEST_PART_SIZE = 0.2


# LOADING THE DATASET
train_set = pd.read_csv(ROOT_PATH / 'train_ship_segmentations_v2.csv')

# MASKS
EMPTY_MASK = train_set[train_set["EncodedPixels"].isna()].sample(EMPTY_IMAGES) # Random set of empty images
SHIPS_MASK = train_set[~train_set["EncodedPixels"].isna()].sample(SHIPS_IMAGES) # Random set of ship images
MASK_DF = pd.concat([EMPTY_MASK, SHIPS_MASK]) # This mask could be used to artificially limit the dataset

# CHECKING FOLDERS
if not MODELS_FOLDER.exists():
    MODELS_FOLDER.mkdir()


# FUNCTIONS
## VISUALIZATION
def visualize(loaded_history: tfk.callbacks.History) -> None:
    '''
    This function visualizes the history of the model
    :loaded_history: tfk.History

    Returns None
    '''
    loss_bce = [] # Orange
    val_loss_bce = [] # Red
    train_dice = [] # Blue
    val_dice = [] # Green

    # Collecting metrics data from every epoch
    # for array in loaded_history:
    loss_bce.extend(loaded_history.history['loss'])
    val_loss_bce.extend(loaded_history.history['val_loss'])
    train_dice.extend(loaded_history.history['dice_score'])
    val_dice.extend(loaded_history.history['val_dice_score'])

    # Defining the plots
    plt.plot(train_dice, label='Training Dice Score', color='blue')
    plt.plot(val_dice, label='Validation Dice Score', color='green')
    plt.plot(loss_bce, label='BCE Loss', color='orange')
    plt.plot(val_loss_bce, label='Validation BCE Loss', color='red')

    # Plot configuration
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Score/Loss')
    plt.title('Training and Validation Metrics')

    plt.show(block=True)


## LOSS
# Defining the loss functions for the model
@tfk.saving.register_keras_serializable()
def dice_score(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Dice score loss function
    :y_true: tf.Tensor
    :y_pred: tf.Tensor
    
    Returns tf.Tensor
    '''
    return (2.0 * K.sum(y_pred * y_true) + 0.0001) / (K.sum(y_true) + K.sum(y_pred) + 0.0001)

@tfk.saving.register_keras_serializable()
def BCE_dice(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    '''
    Combined binary crossentropy and dice score
    :y_true: tf.Tensor
    :y_pred: tf.Tensor
    
    Returns tf.Tensor
    '''
    # Weight
    alpha = 1.0  # BCE weight
    beta = 1.0   # Dice score weight

    bce = K.binary_crossentropy(y_true, y_pred) # BCE
    dice = 1 - dice_score(y_true, y_pred) # Dice score
    loss = alpha * bce + beta * dice # Combining
    return loss


# ENTRY POINT
if __name__ == '__main__':
    # Dataset splitting
    train_df, valid_df = train_test_split(MASK_DF, test_size=TEST_PART_SIZE)

    # Initializing generators
    train = data_generator(TRAIN_FOLDER, BATCH_SIZE, train_df)
    valid = data_generator(TRAIN_FOLDER, BATCH_SIZE, valid_df)

    # Defining checkpoints
    callback = tfk.callbacks.ModelCheckpoint(MODELS_FOLDER / "model.{epoch:02d}-val_loss-{val_loss:.4f}-dice-{dice_score:.4f}.keras", "val_loss", save_best_only=True, save_weights_only=True)

    # Defining the model
    model = unet_model((*train.image_sizes, 3))

    # Loading weights if model is selected
    if MODEL_NAME:
        model.load_weights(MODELS_FOLDER / MODEL_NAME)

    # Load the history if selected
    if HISTORY_NAME:
        # Loading the history
        with open(MODELS_FOLDER / HISTORY_NAME, 'rb') as file:
            loaded_history = pickle.load(file)
            print(loaded_history.history['loss']) # Check
        
    # Compiling the model
    model.compile(tf.keras.optimizers.Adam(ADAM_RATE) , BCE_dice  , dice_score)

    # Training the model
    cur_history = model.fit(train, validation_data=valid, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[callback])

    # Saving the last model weights
    model.save_weights(MODELS_FOLDER / 'last_model_weights.keras')

    # Merging the history if needed
    if loaded_history:
        loaded_history.history['loss'] = loaded_history.history['loss'] + cur_history.history['loss']
        loaded_history.history['val_loss'] = loaded_history.history['val_loss'] + cur_history.history['val_loss']
        loaded_history.history['dice_score'] = loaded_history.history['dice_score'] + cur_history.history['dice_score']
        loaded_history.history['val_dice_score'] = loaded_history.history['val_dice_score'] + cur_history.history['val_dice_score']

    # Save the history file
    with open(MODELS_FOLDER / 'last_model_history.pickle', 'wb') as file:
        pickle.dump(loaded_history, file)

    # Visualizing the history
    visualize(loaded_history)

