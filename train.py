import os
import numpy as np
import tensorflow as tf
from keras.callbacks import ModelCheckpoint
from model import denseNet_model, unet_model
from data_importer import KinoformDataGenerator 

# Define paths and parameters
image_dir = "G:/My Drive/Colab Notebooks/deep_learning_holography_training_1/images/"
hologram_dir = "G:/My Drive/Colab Notebooks/deep_learning_holography_training_1/labels/"
img_size = (64, 64)
batch_size = 32
epochs = 30

# Get input image and hologram paths
image_paths = sorted([
    os.path.join(image_dir, file_name)
    for file_name in os.listdir(image_dir)
    if file_name.endswith(".png")
])
hologram_paths = sorted([
    os.path.join(hologram_dir, file_name)
    for file_name in os.listdir(hologram_dir)
    if file_name.endswith(".png")
])

# Split data into training and validation sets
validate_samples = 100
train_image_paths = image_paths[:-validate_samples]
train_hologram_paths = hologram_paths[:-validate_samples]
validate_image_paths = image_paths[-validate_samples:]
validate_hologram_paths = hologram_paths[-validate_samples:]

# Instantiate data Generators
train_data_generator = KinoformDataGenerator(batch_size, img_size, train_image_paths, train_hologram_paths)  
validate_data_generator = KinoformDataGenerator(batch_size, img_size, validate_image_paths, validate_hologram_paths) 

# Load model from model.py
model = denseNet_model(img_size)

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Define callbacks
callbacks = [tf.keras.callbacks.ModelCheckpoint("DeepLearning_CGH_denseNet.h5", save_best_only=True)]

# Train model
model.fit(train_data_generator, epochs=epochs, validation_data=validate_data_generator, callbacks=callbacks)
