import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

class KinoformDataset:
    def __init__(self, batch_size, image_size, image_paths, hologram_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = image_paths
        self.hologram_paths = hologram_paths

    def __len__(self):
        return len(self.hologram_paths) // self.batch_size # The total number of batches

    def __getitem__(self, idx):
        start_index = idx * self.batch_size # Starting index for the current batch
        # Slice the image and hologram paths to get paths for the current batch
        batch_image_paths = self.image_paths[start_index: start_index + self.batch_size]
        batch_hologram_paths = self.hologram_paths[start_index: start_index + self.batch_size]

        # Initialize arrays to store images and holograms for the batch
        batch_images = np.zeros((self.batch_size,) + self.image_size, dtype="float32")
        batch_holograms = np.zeros((self.batch_size,) + self.image_size, dtype="float32")

        # Load images and holograms for the current batch
        for i, (image_path, hologram_path) in enumerate(zip(batch_image_paths, batch_hologram_paths)):
            # Load and normalize the image (8-bit)
            image = load_img(image_path, target_size=self.image_size, color_mode='grayscale')
            batch_images[i] = img_to_array(image)[:, :, 0] / 255.0  # Normalize the 8-bit image

            # Load and normalize the hologram (8-bit)
            hologram = load_img(hologram_path, target_size=self.image_size, color_mode='grayscale')
            batch_holograms[i] = img_to_array(hologram)[:, :, 0] / 255.0

        return batch_images, batch_holograms