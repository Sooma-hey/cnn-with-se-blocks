from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
from models.se_block import SEBlock
import tensorflow as tf
import random
import numpy as np
from sklearn.utils import check_random_state
from config.config import BaseConfig

# Set random seed
n = BaseConfig.random
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def EnhancedCNN():
    model = Sequential([
        Conv2D(8, (10, 10), padding='same', input_shape=(32, 32, 3)),  # 1st Conv layer
        Conv2D(64, (5, 5), padding='same'),  # 2nd Conv layer
        Conv2D(32, (2, 2), padding='same'),  # 3rd Conv layer
        Activation('relu'),  # Activation layer
        SEBlock(32),  # SE Block
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling

        Conv2D(64, (3, 3), padding='same'),  # 4th Conv layer
        Activation('relu'),  # Activation layer
        SEBlock(64),  # SE Block
        MaxPooling2D(pool_size=(2, 2)),  # Max pooling

        Flatten(),  # Flatten layer
        Dense(32),  # Fully connected layer
        Activation('relu'),  # Activation layer
        Dense(10),  # Output layer
        Activation('softmax')  # Activation layer for classification
    ])
    return model

# Create model instance and print summary
model = EnhancedCNN()
model.summary()
