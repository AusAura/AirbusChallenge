# **Airbus Ship Detection Challenge**

## Abstract

This is a supervised U-Net model for image segmentation, specifically for ship identification on sattelite images.  For loss functions, Dice score and BCE + Dice score metrics are used. As optimizer, uses Adam. Activation - ELU + Sigmoid. Kernel initializer = he_normal.

### Training

To train the model, use train.py file.

It can use original image size, 3x3 and 2x2 crop data generators for better training and performance optimization. They are located in crop.py, and the mode could be selected in the file.

MODELS_FOLDER (for saving models) is created if does not exist.

Leave MODEL_NAME and HISTORY_NAME empty if do not want to load prevously saved weights and history.
Other parameters are necessary.

Train dataset should have the following features:

- Images should be homogenous. Exception is raised if otherwise.
- It was designed to work with any image shape. However, the model was tested with square images only.
- Image shape is taken from the first image file in the folder.
- Dataset should have .csv file with it for supervised learning.
- The file should have 'ImageId' and 'EncodedPixels' columns.
- Each row should have: 'NaN' if image does not have a ship; single RLE encoded mask for 1 ship.
- Each image can have multiple masks in different rows.

You can use mask to limit the dataset by specifying EMPTY_IMAGES and SHIPS_IMAGES.

Model splits dataset on train and validation parts.
Creates checkpoints with best results
Supports loading previously trained version of the model with the history and allows continue the training
In the end, saves last version of the model with the history
Then visualizes the complete history of training (with previous history)

### Predicting

For predictions, use predict.py file.

MODEL_NAME is necessary.
Output folder is created if does not exist.

Dataset should have the following features:

- Images should be homogenous. Exception is raised if otherwise.
- It was designed to work with any image shape. However, the model was tested with square images only.
- Image shape is taken from the first image file in the folder.
- Model does not require you to create a .csv file but it is using 'sample_submission_v2.csv' to create similar structure for results file.
- You can create and select different save functions with 'saver'. Basic option - saving to .csv file similar to 'train_ship_segmentations_v2.csv'.
- Dataset could be limited with 'SAMPLES_LIMIT'.
- Each predicted mask is blended with the original image and saves in the output folder for furhter visual analysis.
- As the next step, model divides different objects on the predicted masks.
- Each single mask is saved in the results.csv file then as separate row.

## EDA of Basic Dataset

For detailed analysis, please check the EDA.ipybn file. Short version could be found below.

### Results

#### Datatypes

Our analysis shows that columns have np 'object'type which should not be an issue.
In NumPy, 'NaN' is a float.
Values with masks have 'string' type.

#### Image quantity:
1. **Train set:**
- Total images - 192 556
- Total rows (empty, single mask, multiple mask) - 231 723
- Total masks (single, multiple masks) - 81 723
- Total unique images with masks (multiple masks does not count) - 42 556
- Without ships (w/o masks) - 150 000

2. **Sample set:**

Total images - 15 606

3. **Total number:** 208 162


As we can see, there are 42556 images with 81723 ships on them.
It means that roughly 1/4 of the dataset has masks (with ships).

#### Visual analysis:
By running the code a few times, we can learn that some images seem to not detect a few ships. However, false-positives are not present or rare.

## Image processing

### RLE Decoder

Since masks from the train dataset are RLE encoded, it is necessary to decode them before training. For that, RLE decoder is built-in to the data generator.

### Crop and Data Generators

A generator that will apply our crop function to the images.

Typically, when training models for object segmentation tasks (such as ship detection), it is beneficial to retain only the part of the mask that contains the main object to reduce noise and simplify the task for the model.

This approach can also help improve the model's generalization ability since it will be trained on more specific and informative mask data.

Furthermore, it allows us to train the model with limited computational resources.

Imperically was tested that dividing it 2x2 is less effective than 3x3.

Available modes:

1. Crop2x2Generator
2. Crop3x3Generator
3. NoCropGenerator

Use 'data_generator' in crop.py file to select one.

## U-Net Model
### Abstract

The U-Net model is a convolutional neural network architecture primarily used for image segmentation tasks, especially in the medical imaging field. It was introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox in 2015.

The architecture of the U-Net model is characterized by a U-shaped structure, which consists of a contracting path and an expansive path.

The model structure is placed in model.py file.

### Modifications

This CNN is based on classic U-Net with small changes like:
- Decreased channels == features
- Decreased dropout rate
- Since it is a binary classification case, we will use Sigmoid instead of Softmax
- Consequentially, we will take 'he_normal' instead of 'glorot_uniform' for activation that is more suited for Sigmoid
- Using ELU instead ReLU.
- Added BatchOptimization

### Decision Grounding

Functional API for Keras was selected for code-reading convenience and compactness.

1. **ELU (Exponential Linear Unit)**

The drawback of ReLU is the "dying ReLU" problem, where the activation becomes negative and remains unchanged during training. However, ELU is also a nonlinear activation function that overcomes the drawbacks of ReLU. ELU retains all the positive aspects of ReLU and reduces the problem of "dying ReLU". ELU can be particularly useful in networks where the tendency for neurons to die is a problem, such as in deep convolutional neural networks.


2. **He Initialization (he_normal)**

This weight initialization method is based on the work of Kaiming He and is intended for networks with activation functions that suffer from saturation issues, such as the Rectified Linear Unit (ReLU). He initialization is recommended for networks with activation functions like ReLU, which may suffer from the "dying ReLU" problem or 'dead neurons'.

3.  **Dropout**

Dropout is an effective regularization technique that improves the generalization ability of neural networks by preventing overfitting and encouraging the learning of robust features. It is especially useful when dealing with large and complex neural networks or datasets where overfitting is a common concern.

4. **BatchNormalization**

In U-Net architecture, BatchNormalization is also usually applied after activation layers. This is a common practice that helps stabilize training and accelerate model convergence. In addition to stabilization, BatchNormalization can also speed up the training process by reducing the need for lower learning rates and helping to avoid issues such as gradient vanishing.

#### Parameters:

1. **Dropout** = 0.2

Higher dropout (about 0.5) resulted in 'dead neurons' issue.

2. **batch_size** = 16

Lower batch size increases the model's ability to generalize.

3. **Crop** = 3x3

3x3 showed increased accuracy comparing to 2x2 option.

4. **Adam learning rates** = 0.001 + 0.0005 + 0.0001

Taking the recommended values since Adam can optimize its learning rate by itself.

5. **Kf (coefficient, quantity of filters)** = 0.25

Kf is a custom parameter that is used to easily regulate quantity of filters on each layer.
Coefficient 0.25 showed a bit faster results than 0.5. However, it is not completely clear at the moment which is better for accuracy.
Coefficiting 1 caused 'dead neurons' issue.

### Loss Functions

For scoring, two loss functions were tested: dice score and BCE + Dice Loss combination.

1. **Dice Score**

The Dice Score, also known as the Dice Coefficient or Dice Similarity Coefficient, is a measure commonly used to evaluate the performance of image segmentation algorithms, particularly in medical image analysis and computer vision tasks.

It is calculated as the ratio of twice the intersection of the predicted segmentation mask and the ground truth mask to the sum of the number of pixels in both masks.

2. **BCE + Dice Loss**

BCE (Binary Cross-Entropy) loss is used to assess the discrepancy between predicted and true pixel labels, while Dice Loss is used to measure the similarity between segmented regions. Combining these loss functions can help the model achieve a better balance between accuracy and trainability.

For now, only Dice Score is selected.

Loss functions are placed in train.py file.

## Model Compilation and Dataset Splitting

### Splitting our set on train and validation parts, connecting original images with the data generator and defining our model

In train.py, the following parameters are necesarry for model training.

EPOCHS = 2
ADAM_RATE = 0.001
BATCH_SIZE = 16
TEST_PART_SIZE = 0.2

For checkpoints, you can change 'callback' variable.

Metrics could be changed during the model compilation.

### Training

You can continue to train your previously saved model.

## Evaluating the Results

These functions can be used for results evaluation.

![Plot with the results](results.png)

### Result

After the 11th epoch with full dataset, metric values:

**6480s 561ms/step - loss: 0.3290 - dice_score: 0.6909 - val_loss: 0.3518 - val_dice_score: 0.6697**

Validation dice score has increased from 47% to 67%, validation loss has decreased from 56% to 35%. Model have not reached its peak yet.

This can be considered as a good start, we can assume that additional 17-20 epochs will be enough to reach the maximum result. Further tests from my side are not possible.

Visual analysis of blended images confirms shown accuracy.