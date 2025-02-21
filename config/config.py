# config/config.py

class BaseConfig:
    # Base parameters
    batch_size = 100
    num_workers = 2
    random = 3
    data_dir = 'data/cifar-10-batches-py'
class BaselineConfig(BaseConfig):
    # Model parameters for baseline model
    num_epochs = 30
    learning_rate = 0.001
    # Model save path
    checkpoint_path = 'results/checkpoints/baseline_model/best_model.h5'

class EnhancedConfig(BaseConfig):
    # Model parameters for enhanced model
    num_epochs = 30
    learning_rate = 0.001
    # Model save path
    checkpoint_path = 'results/checkpoints/enhanced_model/best_model.h5'
