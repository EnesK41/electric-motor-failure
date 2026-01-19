import scipy.io
import torch
import numpy as np
import scipy.signal
from transformers import ASTFeatureExtractor, ASTModel

# 1. Configuration
# ---------------------------------------------------------
# CWRU data is 12k or 48k Hz. AST model expects 16k Hz.
TARGET_SAMPLE_RATE = 16000 
INPUT_FILE = 'data/raw/105.mat'

# 2. Model Initialization
# ---------------------------------------------------------
print("Downloading model from Hugging Face...")
feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
print("Model ready.")

# 3. Data Loading and Preprocessing
# ---------------------------------------------------------
def load_and_preprocess(file_path):
    # a. Read file
    mat = scipy.io.loadmat(file_path)
    
    # b. Find signal (DE_time, FE_time etc.)
    signal = None
    for key in mat.keys():
        if 'DE_time' in key: # Get Drive End signal
            signal = mat[key].flatten() # Flatten to 1D array
            break
            
    if signal is None:
        raise ValueError("Signal not found!")

    # c. Split signal into 1-second chunks (Taking the first 1 second for example)
    # For CWRU 12k data, 1 second = 12000 points
    original_sr = 12000 
    one_second_signal = signal[:original_sr] 

    # d. RESAMPLING (12k -> 16k)
    # AST expects 16,000 Hz. Resample correctly.
    # Mathematical ratio: 16000 / 12000
    num_samples_target = int(len(one_second_signal) * TARGET_SAMPLE_RATE / original_sr)
    resampled_signal = scipy.signal.resample(one_second_signal, num_samples_target)
    
    return resampled_signal

# 4. Inference
# ---------------------------------------------------------
print("Processing signal...")
raw_audio = load_and_preprocess(INPUT_FILE)

# Convert data to model input format
inputs = feature_extractor(raw_audio, sampling_rate=TARGET_SAMPLE_RATE, return_tensors="pt")

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

# 5. Results
# ---------------------------------------------------------
# last_hidden_state: Sequence of hidden states at the last layer of the model.
# pooler_output: Last layer hidden-state of the first token of the sequence (classification token).
last_hidden_state = outputs.last_hidden_state
pooler_output = outputs.pooler_output

print("Embeddings extracted successfully.")
print(f"Detailed Output Size (Sequence): {last_hidden_state.shape}") 
print(f"Summary Embedding Size (Vector):  {pooler_output.shape}")
print("Example (First 10 values):", pooler_output[0, :10].numpy())