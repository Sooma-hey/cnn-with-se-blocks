import tensorflow as tf
from tensorflow.keras.layers import Layer, GlobalAveragePooling2D, Dense, Reshape, Multiply, BatchNormalization, Activation
from config.config import BaseConfig
import random
import numpy as np
from sklearn.utils import check_random_state

# Set random seed
n = BaseConfig.random
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

class SEBlock(Layer):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        # Define layers for SE Block
        self.global_avg_pool = GlobalAveragePooling2D()
        self.fc1 = Dense(channels // reduction, use_bias=False)
        self.bn1 = BatchNormalization()
        self.act1 = Activation('relu')
        self.fc2 = Dense(channels // reduction, use_bias=False)
        self.bn2 = BatchNormalization()
        self.act2 = Activation('relu')
        self.fc3 = Dense(channels, use_bias=False)
        self.act3 = Activation('sigmoid')
        self.reshape = Reshape((1, 1, channels))

    def call(self, inputs):
        # Forward pass through SE Block
        x = self.global_avg_pool(inputs)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        x = self.act3(x)
        x = self.reshape(x)
        return Multiply()([inputs, x])
