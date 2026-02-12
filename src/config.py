import os
import torch

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Point to the fine-tuned model if available, otherwise fallback
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "motor_model_best_tuned.pth")
if not os.path.exists(MODEL_SAVE_PATH):
    MODEL_SAVE_PATH = os.path.join(BASE_DIR, "motor_model_best.pth")

# Model Configuration
ENCODER_ID = "MIT/ast-finetuned-audioset-10-10-0.4593"
DECODER_ID = "gpt2"
ENCODER_DIM = 768
DECODER_DIM = 768

# Data Processing
SAMPLE_RATE = 16000
CHUNK_SIZE = 12000
STEP_SIZE = 12000  # Non-overlapping
SEQ_LEN = 1024     # AST input sequence length

# Training Hyperparameters
BATCH_SIZE = 4
LEARNING_RATE = 1e-5
EPOCHS = 20
PATIENCE = 5
MAX_GENERATION_LEN = 20
