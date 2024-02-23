import os
import numpy as np
from keras_preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence

class KinoformDataGenerator(Sequence):
    def __init__(self, batch_size, image_size, image_paths, hologram_paths):
        self.batch_size = batch_size
        self.image_size = image_size
        self.image_paths = image_paths
        self.hologram_paths = hologram_paths
        self.n = len(hologram_paths)

    def __len__(self):
        return self.n // self.batch_size

    def __getitem__(self, idx):
        batch_image_paths = self.image_paths[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_hologram_paths = self.hologram_paths[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_images = np.zeros((self.batch_size,) + self.image_size, dtype="float32")
        batch_holograms = np.zeros((self.batch_size,) + self.image_size, dtype="float32")

        for i, (image_path, hologram_path) in enumerate(zip(batch_image_paths, batch_hologram_paths)):
            image = load_img(image_path, target_size=self.image_size, color_mode='grayscale')
            batch_images[i] = img_to_array(image)[:, :, 0] / 255.0

            hologram = load_img(hologram_path, target_size=self.image_size, color_mode='grayscale')
            batch_holograms[i] = img_to_array(hologram)[:, :, 0] / 255.0

        return batch_images, batch_holograms
