import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from model import denseNet_model, unet_model
from data_importer import KinoformDataset

# Define paths and parameters
image_dir = "G:/My Drive/Colab Notebooks/deep_learning_holography_training_1/images/"
hologram_dir = "G:/My Drive/Colab Notebooks/deep_learning_holography_training_1/labels/"
model_output_dir = "trained_models"  # diretory to save trained models
img_size = (64, 64)
batch_size = 32
epochs = 10

# Get input and target image paths
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
train_target_img_paths = hologram_paths[:-validate_samples]
val_input_img_paths = image_paths[-validate_samples:]
val_target_img_paths = image_paths[-validate_samples:]

# Instantiate data Sequences
train_gen = KinoformDataset(batch_size, img_size, train_image_paths, train_target_img_paths)
val_gen = KinoformDataset(batch_size, img_size, val_input_img_paths, val_target_img_paths)

# Load model from model.py
model = denseNet_model(img_size)

# Compile model
model.compile(optimizer="adam", loss="mean_squared_error")

# Define callbacks
callbacks = [ModelCheckpoint(os.path.join(model_output_dir, "DeepLearning_CGH_denseNet.h5"), save_best_only=True)]

# Train model
model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)