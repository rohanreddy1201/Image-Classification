# Image Classification using Tensorflow on the Intel Image dataset

This is my implementation of classifying the images provided in the intel image dataset and providing a statistical result.

# Dataset
The dataset can be downloaded at: https://www.kaggle.com/datasets/puneet6060/intel-image-classification
This Data contains around 25k images of size 150x150 distributed under 6 categories.
{'buildings' -> 0,
'forest' -> 1,
'glacier' -> 2,
'mountain' -> 3,
'sea' -> 4,
'street' -> 5 }

The Train, Test and Prediction data is separated in each zip files. There are around 14k images in Train, 3k in Test and 7k in Prediction.

# Explanation
Importing Libraries:  
numpy, os, warnings: Standard libraries for numerical operations, file system operations, and managing warnings respectively.  
matplotlib.pyplot: Library for creating plots and visualizations.  
ImageDataGenerator, load_img, img_to_array: Utilities from TensorFlow's Keras module for data augmentation and image processing.  
Sequential, Conv2D, MaxPooling2D, Flatten, Dense: Components from TensorFlow's Keras module for building convolutional neural network (CNN) models.  

Setting Up Data Directories and Parameters:  
train_data_dir, test_data_dir, pred_data_dir: Paths to directories containing training, testing, and prediction images respectively.  
img_width, img_height: Dimensions for resizing images.  
input_shape: Shape of input images to the CNN model.  
epochs, batch_size: Parameters controlling training duration and batch size.  

Data Augmentation and Normalization:  
ImageDataGenerator: Instances for augmenting and normalizing image data.  
train_datagen, test_datagen: Image data generators for training and testing/validation sets respectively.  
train_generator, validation_generator: Flow generators for iterating over images in batches during training and validation.  

Defining the CNN Model:
Sequential: A linear stack of layers for building the model.  
Conv2D: Convolutional layers for feature extraction.  
MaxPooling2D: Pooling layers for downsampling and feature selection.  
Flatten: Layer for flattening the output of convolutional layers into a 1D array.  
Dense: Fully connected layers for classification.  

Compiling the Model:
compile: Configuring the model for training by specifying loss function, optimizer, and evaluation metrics.  

Training the Model:
fit: Training the model on training data and validating on validation data for a specified number of epochs.  

Plotting Training and Validation Metrics:
Using matplotlib.pyplot to visualize training and validation accuracy and loss over epochs.  

Predicting on New Images:
Iterating over images in the prediction directory, loading each image, preprocessing it, and making predictions using the trained model.
Collecting the predicted classes.  

Plotting Distribution of Predicted Classes:
Visualizing the distribution of predicted classes using a bar plot.

## Disclaimers
This project is for machine learning research and educational purposes only. This cannot be used for professional advice and application.
