"""
Configuration file for OOB Detection
Contains all constants and hyperparameters
"""

# Random seed for reproducibility
RANDOM_SEED = 1

# Jersey color mapping for each clip
JERSEY_COLORS = {
    'clip 1': 'white',
    'clip 2': 'white',
    'clip 3': 'blue',
    'clip 4': 'blue',
    'clip 5': 'blue',
    'clip 6': 'blue',
    'clip 7': 'blue',
    'clip 8': 'blue',
    'clip 9': 'blue',
    'clip 10': 'black',
    'clip 11': 'black',
    'clip 12': 'black',
    'clip 13': 'black',
    'clip 14': 'blue',
    'clip 15': 'blue',
    'clip 16': 'blue',
    'clip 17': 'blue',
    'clip 18': 'white',
    'clip 19': 'black',
    'clip 20': 'black'
}

# Color to one-hot encoding
COLOR_ENCODING = {
    'white': [1, 0, 0],
    'blue': [0, 1, 0],
    'black': [0, 0, 1]
}

# Model hyperparameters
NUM_FRAMES = 8
BATCH_SIZE = 4
EPOCHS = 5
LEARNING_RATE = 0.001
LSTM_HIDDEN_SIZE = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
COLOR_EMBEDDING_SIZE = 16

# Training parameters
LR_SCHEDULER_PATIENCE = 2
LR_SCHEDULER_FACTOR = 0.5

# Cross-validation
N_FOLDS = 4

# Data augmentation
AUGMENTATION_MULTIPLIER = 3
HORIZONTAL_FLIP_PROB = 0.5
COLOR_JITTER_BRIGHTNESS = 0.2
COLOR_JITTER_CONTRAST = 0.2

# Image normalization (ImageNet stats)
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Paths
VIDEO_DIR = '/content/'
VIDEO_PATTERN = 'clip*.mp4'
