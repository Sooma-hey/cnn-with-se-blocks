from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from models.baseline_cnn import BaselineCNN
from models.enhanced_cnn import EnhancedCNN
from config.config import BaselineConfig, EnhancedConfig, BaseConfig
import tensorflow as tf
import random
import numpy as np
from sklearn.utils import check_random_state

# Setting random seed for reproducibility
n = BaseConfig.random
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def train_model(train_data, val_data, model_type='enhanced', patience=5):
    if model_type == 'baseline':
        config = BaselineConfig
        model = BaselineCNN()
    else:
        config = EnhancedConfig
        model = EnhancedCNN()

    # Compile the model with Adam optimizer and sparse categorical crossentropy loss
    model.compile(optimizer=Adam(learning_rate=config.learning_rate),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    # Set up early stopping and model checkpoint callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint(f'{config.checkpoint_path}', save_best_only=True, monitor='val_loss')

    # Train the model with training and validation data
    history = model.fit(train_data, validation_data=val_data, epochs=config.num_epochs,
                        callbacks=[early_stopping, model_checkpoint])

    print("Training completed.")
    return model, history
