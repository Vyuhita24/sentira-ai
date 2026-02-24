import os

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_PATH = os.path.join(BASE_DIR, "stress_model.keras")

# Audio Config
SAMPLE_RATE = 22050
DURATION = 3 # Seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

# Feature Extraction
N_MFCC = 40
HOP_LENGTH = 512
N_FFT = 2048

# Model Config
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.0001
Validation_Split = 0.2
