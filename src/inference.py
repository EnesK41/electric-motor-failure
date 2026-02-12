import torch
import random
from transformers import ASTFeatureExtractor, GPT2Tokenizer
from model import SignalCaptioningModel
from dataset import CWRUDataset
import config
import utils

def run_inference():
    print(f"Inference Device: {config.DEVICE}")

    # Initialize Components
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.ENCODER_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(config.DECODER_ID)
    
    model = SignalCaptioningModel(config.ENCODER_ID, config.DECODER_ID).to(config.DEVICE)

    # Load Model
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print(f"Model loaded from {config.MODEL_SAVE_PATH}")
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    model.eval()

    # Load Dataset (for demo purposes)
    print("Loading dataset for example generation...")
    dataset = CWRUDataset(config.RAW_DATA_DIR, feature_extractor)

    print("\n--- Manual Inference Demo ---\n")

    for i in range(5):
        idx = random.randint(0, len(dataset)-1)
        input_values, true_label = dataset[idx]
        input_tensor = input_values.unsqueeze(0).to(config.DEVICE)

        print(f"Example {i+1} (ID: {idx})")
        print(f"Ground Truth: {true_label}")
        
        # Generate Caption using Utility Function
        caption = utils.generate_caption(
            model, 
            tokenizer, 
            input_tensor, 
            max_len=config.MAX_GENERATION_LEN, 
            device=config.DEVICE
        )

        print(f"Prediction:   {caption}")
        
        # Basic Validation
        if true_label.lower() in caption.lower() or caption.lower() in true_label.lower():
            print("Status: SUCCESS")
        else:
            print("Status: FAILURE")
        print("\n" + "-"*30 + "\n")

if __name__ == "__main__":
    run_inference()