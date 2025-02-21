import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from config.config import BaseConfig
import pickle
import tensorflow as tf
import random
import numpy as np
from sklearn.utils import check_random_state

# Set random seed
n = BaseConfig.random
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def load_cifar10_batch(batch_filename):
    # Load CIFAR-10 batch data
    with open(batch_filename, 'rb') as f:
        batch = pickle.load(f, encoding='latin1')
    data = batch['data']
    labels = batch['labels']
    data = data.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float32")
    return data, labels

def load_train_val_data(val_split=0.1):
    data_dir = BaseConfig.data_dir
    train_data = []
    train_labels = []

    # Load and concatenate all training batches
    for i in range(1, 6):
        data_batch, labels_batch = load_cifar10_batch(os.path.join(data_dir, f'data_batch_{i}'))
        train_data.append(data_batch)
        train_labels.append(labels_batch)

    train_data = np.concatenate(train_data)
    train_labels = np.concatenate(train_labels)

    # Normalize the images
    train_data = train_data / 255.0

    # Split training data into train and validation sets
    x_train, x_val, y_train, y_val = train_test_split(train_data, train_labels, test_size=val_split)

    # Define data augmentation for training data
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
    )

    # Define data generators for training and validation data
    train_generator = train_datagen.flow(x_train, y_train, batch_size=BaseConfig.batch_size)
    val_generator = ImageDataGenerator().flow(x_val, y_val, batch_size=BaseConfig.batch_size)

    return train_generator, val_generator

def load_test_data():
    data_dir = BaseConfig.data_dir
    # Load CIFAR-10 test data
    test_data, test_labels = load_cifar10_batch(os.path.join(data_dir, 'test_batch'))

    # Normalize the images
    test_data = test_data / 255.0

    # Define data generator for test data
    test_generator = ImageDataGenerator().flow(test_data, test_labels, batch_size=BaseConfig.batch_size)

    return test_generator
