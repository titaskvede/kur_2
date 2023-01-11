# This file is responsible for realising neural network and predicting the results
import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import keras

class Neural:
    def __init__(self, data=os.path.join(os.getcwd(), "run_0")):
        self.data = []
        self.data_dir = data
        self.img_dim = [None, None]
        self.batch_size = 32

    def training(self):

        train_ds = tensorflow.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=(self.img_dim[0], self.img_dim[1]),
            batch_size=self.batch_size)

        val_ds = tensorflow.keras.preprocessing.image_dataset_from_directory(
            self.data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=(self.img_dim[0], self.img_dim[1]),
            batch_size=self.batch_size)

        # Use buffered prefetching to load images from disk without having I/O become blocking
        AUTOTUNE = tensorflow.data.AUTOTUNE

        train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

        normalization_layer = Rescaling(1. / 255)
        normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
        image_batch, labels_batch = next(iter(normalized_ds))
        first_image = image_batch[0]

        class_names = os.listdir(os.path.join(os.getcwd(), 'run_0'))
        num_classes = len(class_names)

        # Model
        model = tensorflow.keras.Sequential([
            Rescaling(1. / 255, input_shape=(self.img_dim[0], self.img_dim[1], 3)),
            layers.Conv2D(64, 1, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 1, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(16, 1, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(num_classes)
        ])

        model.compile(optimizer='adam',
                      loss=tensorflow.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=['accuracy'])

        model.summary()

        epochs = 10
        history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=epochs
        )

        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(epochs)

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

        model.save(os.path.join("data", "created_model", "final_modelll"))

    def check_random_image_and_set_dimensions(self, data_folder=os.path.join("0", "*"), **kwargs):
        images = list(pathlib.Path(self.data_dir).glob(data_folder))
        image = PIL.Image.open(str(images[0]))
        if "show" in kwargs:
            image.show()
        self.img_dim = image.size

    def collect_all_data(self):
        for data in os.listdir(self.data_dir):
            self.data.append(data)

    def main(self):
        self.collect_all_data()
        self.check_random_image_and_set_dimensions(data_folder=os.path.join("hairpin", "*"))
        self.training()
        pass


if __name__ == "__main__":
    neural = Neural()
    neural.main()
    neural.check_random_image_and_set_dimensions(data_folder=os.path.join("hairpin", "*"))
    pass
