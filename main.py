import tensorflow as tf
from scripts.train import train_model
from scripts.evaluate import evaluate_model
from scripts.utils import load_train_val_data, load_test_data
import random
import numpy as np
from sklearn.utils import check_random_state
from config.config import BaseConfig, BaselineConfig, EnhancedConfig
import argparse
import matplotlib.pyplot as plt
import os

# Set random seed
n = BaseConfig.random
np.random.seed(n)
random.seed(n)
tf.random.set_seed(n)
check_random_state(n)

def get_paths(model_type):
    # Get checkpoint path based on model type
    if model_type == 'baseline':
        return BaselineConfig.checkpoint_path
    elif model_type == 'enhanced':
        return EnhancedConfig.checkpoint_path
    else:
        raise ValueError("Invalid model type. Choose 'baseline' or 'enhanced'.")

def save_plots(history, model_type):
    os.makedirs(f'results/plots/{model_type}', exist_ok=True)

    # Plot and save accuracy
    plt.figure()
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.savefig(f'results/plots/{model_type}/accuracy.png', dpi=100)
    plt.show()

    # Plot and save loss
    plt.figure()
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(f'results/plots/{model_type}/loss.png', dpi=100)
    plt.show()

def main(mode=None, model_type=None):
    if mode is None:
        mode = input("Enter mode (train/evaluate): ").strip().lower()

    if model_type is None:
        model_type = input("Enter model type (baseline/enhanced): ").strip().lower()

    model_path = get_paths(model_type)

    if mode == 'train':
        # Load train and validation data
        train_data, val_data = load_train_val_data()

        # Train model
        model, history = train_model(train_data, val_data, model_type=model_type)

        # Save and display plots
        save_plots(history, model_type)

    elif mode == 'evaluate':
        if model_path:
            # Load test data
            test_data = load_test_data()

            # Evaluate the model
            evaluate_model(model_path, test_data, model_type)
        else:
            print("Please provide the model type for evaluation.")
    else:
        print("Invalid mode. Please choose either 'train' or 'evaluate'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="Mode: 'train' or 'evaluate'")
    parser.add_argument("--model_type", help="Model type: 'baseline' or 'enhanced'")
    args = parser.parse_args()

    main(args.mode, args.model_type)
