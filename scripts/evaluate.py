from models.se_block import SEBlock
import tensorflow as tf

def evaluate_model(model_path, test_data, model_type):
    if model_type == 'enhanced':
        # Load the enhanced model with custom objects
        model = tf.keras.models.load_model(model_path, custom_objects={'SEBlock': SEBlock})
    else:
        # Load the baseline model without custom objects
        model = tf.keras.models.load_model(model_path)

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_data)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")
