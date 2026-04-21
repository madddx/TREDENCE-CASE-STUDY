"""
Configuration and Constants for Self-Pruning Neural Network
"""

# ============================================================================
# DATASET CONFIGURATION
# ============================================================================

DATASET_NAME = "CIFAR-10"
DATA_PATH = "./data"
NUM_CLASSES = 10

# CIFAR-10 image properties
IMAGE_SIZE = 32  # 32x32 pixels
NUM_CHANNELS = 3  # RGB
FLATTENED_INPUT_SIZE = IMAGE_SIZE * IMAGE_SIZE * NUM_CHANNELS  # 3072

# Data augmentation
ENABLE_DATA_AUGMENTATION = True
NORMALIZATION_MEAN = (0.4914, 0.4822, 0.4465)
NORMALIZATION_STD = (0.2023, 0.1994, 0.2010)

# ============================================================================
# NETWORK ARCHITECTURE
# ============================================================================

INPUT_SIZE = FLATTENED_INPUT_SIZE  # 3072 for CIFAR-10
HIDDEN_SIZES = [512, 256]  # Two hidden layers
OUTPUT_SIZE = NUM_CLASSES  # 10 for CIFAR-10

# Alternative architectures to try:
# Smaller network:     [256, 128]
# Deeper network:      [512, 256, 128]
# Wider network:       [1024, 512, 256]
# Very deep:           [512, 256, 128, 64]

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Batch size
BATCH_SIZE = 128  # Reduce to 64 if out of memory

# Optimizer
OPTIMIZER_NAME = "Adam"
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5

# Learning rate scheduler
SCHEDULER_NAME = "CosineAnnealing"
SCHEDULER_T_MAX = 30  # Cosine annealing period

# Training duration
NUM_EPOCHS = 30
EARLY_STOPPING_PATIENCE = None  # Disable early stopping

# ============================================================================
# SPARSITY CONFIGURATION
# ============================================================================

# Lambda values: control sparsity-accuracy trade-off
# Lower λ → less pruning, higher accuracy
# Higher λ → more pruning, lower accuracy
LAMBDA_VALUES = [
    0.0001,  # Light pruning (expect ~12% sparsity)
    0.001,   # Medium pruning (expect ~29% sparsity)
    0.01,    # Heavy pruning (expect ~46% sparsity)
]

# Alternative lambda values to experiment:
# LAMBDA_VALUES = [0.00001, 0.0001, 0.001, 0.01, 0.1]  # Finer sweep
# LAMBDA_VALUES = [0.0005]  # Single value for quick test

# Gate threshold for sparsity calculation
# Gates below this value are considered "pruned"
SPARSITY_THRESHOLD = 1e-2  # 0.01

# Initial gate score
# Higher values → more gates start active
# Lower values → more gates start inactive
INITIAL_GATE_SCORE = 0.5  # sigmoid(0.5) ≈ 0.73 activation

# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

# Metric calculation
CALCULATE_SPARSITY_EVERY_EPOCH = True
CALCULATE_LAYER_SPARSITY = True

# Checkpoint saving
SAVE_BEST_MODEL = True
SAVE_CHECKPOINTS_EVERY_N_EPOCHS = 10  # Set to None to disable

# ============================================================================
# VISUALIZATION CONFIGURATION
# ============================================================================

# Plot settings
PLOT_DPI = 300
PLOT_STYLE = "seaborn-v0_8-darkgrid"
FIGURE_SIZE = (14, 10)

# Gate distribution histogram
GATE_HISTOGRAM_BINS = 100
GATE_DISTRIBUTION_ALPHA = 0.7
GATE_DISTRIBUTION_COLOR = "steelblue"

# Training curves
SHOW_TRAINING_CURVES = True
SAVE_TRAINING_CURVES = True

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Results directory
RESULTS_DIR = "./results"
CREATE_RESULTS_DIR = True

# Output files
SAVE_RESULTS_JSON = True
SAVE_RESULTS_TABLE = True
SAVE_GATE_DISTRIBUTIONS = True
SAVE_TRAINING_CURVES = True

# Logging
VERBOSE = True
PROGRESS_BAR = True

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

# Auto-detect: 'cuda' if available, else 'cpu'
DEVICE = "auto"

# Or explicitly set:
# DEVICE = "cuda"
# DEVICE = "cpu"

# Number of workers for DataLoader
NUM_WORKERS = 2  # Set to 0 if on Windows or having issues

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42  # Set to None for non-deterministic behavior
DETERMINISTIC = True  # May slow down training slightly

# ============================================================================
# ADVANCED CONFIGURATION
# ============================================================================

# Gate initialization strategy
GATE_INIT_STRATEGY = "constant"  # Options: "constant", "normal", "uniform"

# Gradient clipping
CLIP_GRADIENTS = False
GRAD_CLIP_VALUE = 1.0

# Dropout (if added to network)
DROPOUT_RATE = 0.0  # Set > 0 to enable dropout

# Loss function weights
CLASSIFICATION_LOSS_WEIGHT = 1.0
SPARSITY_LOSS_WEIGHT = "lambda"  # Use lambda value

# ============================================================================
# EXPERIMENT PROFILES
# ============================================================================

# Predefined profiles for quick experimentation

PROFILES = {
    "quick_test": {
        "NUM_EPOCHS": 5,
        "BATCH_SIZE": 256,
        "LAMBDA_VALUES": [0.001],
        "HIDDEN_SIZES": [256, 128],
    },
    "standard": {
        "NUM_EPOCHS": 30,
        "BATCH_SIZE": 128,
        "LAMBDA_VALUES": [0.0001, 0.001, 0.01],
        "HIDDEN_SIZES": [512, 256],
    },
    "thorough": {
        "NUM_EPOCHS": 50,
        "BATCH_SIZE": 64,
        "LAMBDA_VALUES": [0.00001, 0.0001, 0.001, 0.01, 0.1],
        "HIDDEN_SIZES": [512, 256, 128],
    },
    "deep_network": {
        "NUM_EPOCHS": 40,
        "BATCH_SIZE": 128,
        "LAMBDA_VALUES": [0.001],
        "HIDDEN_SIZES": [512, 256, 128, 64],
    },
}

# Use a profile:
# python pruning_network.py --profile thorough

# ============================================================================
# FUNCTION TO LOAD CONFIG
# ============================================================================

def load_profile(profile_name: str) -> dict:
    """
    Load a configuration profile.
    
    Args:
        profile_name: Name of the profile (e.g., 'standard', 'quick_test')
        
    Returns:
        Dictionary with configuration values
    """
    if profile_name not in PROFILES:
        raise ValueError(f"Unknown profile: {profile_name}. Available: {list(PROFILES.keys())}")
    
    return PROFILES[profile_name]


def get_config() -> dict:
    """
    Get current configuration as dictionary.
    
    Returns:
        Dictionary with all configuration values
    """
    return {
        # Dataset
        "DATASET_NAME": DATASET_NAME,
        "DATA_PATH": DATA_PATH,
        "NUM_CLASSES": NUM_CLASSES,
        "FLATTENED_INPUT_SIZE": FLATTENED_INPUT_SIZE,
        
        # Network
        "HIDDEN_SIZES": HIDDEN_SIZES,
        "INPUT_SIZE": INPUT_SIZE,
        "OUTPUT_SIZE": OUTPUT_SIZE,
        
        # Training
        "BATCH_SIZE": BATCH_SIZE,
        "LEARNING_RATE": LEARNING_RATE,
        "NUM_EPOCHS": NUM_EPOCHS,
        
        # Sparsity
        "LAMBDA_VALUES": LAMBDA_VALUES,
        "SPARSITY_THRESHOLD": SPARSITY_THRESHOLD,
        
        # Device
        "DEVICE": DEVICE,
        "NUM_WORKERS": NUM_WORKERS,
    }


if __name__ == "__main__":
    # Print current configuration
    import json
    
    config = get_config()
    print(json.dumps(config, indent=2))
