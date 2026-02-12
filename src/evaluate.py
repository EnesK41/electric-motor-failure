import torch
from tqdm import tqdm
from transformers import ASTFeatureExtractor, GPT2Tokenizer
from model import SignalCaptioningModel
from dataset import CWRUDataset
import config
import utils

def evaluate():
    print(f"Test Device: {config.DEVICE}")

    # Load Model components
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.ENCODER_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(config.DECODER_ID)
    tokenizer.pad_token = tokenizer.eos_token

    model = SignalCaptioningModel(config.ENCODER_ID, config.DECODER_ID).to(config.DEVICE)
    
    try:
        model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Error: Model file not found.")
        exit()

    model.eval()

    # Load Dataset
    print("Preparing dataset (loading all samples)...")
    dataset = CWRUDataset(config.RAW_DATA_DIR, feature_extractor)
    total_samples = len(dataset)
    
    print(f"\nEvaluating on {total_samples} samples...")

    # Metrics
    exact_matches = 0       
    fault_type_matches = 0  
    fault_size_matches = 0  
    errors = []

    for idx in tqdm(range(total_samples), desc="Running Inference"):
        input_values, true_label = dataset[idx]
        input_tensor = input_values.unsqueeze(0).to(config.DEVICE)

        # Generate Caption
        prediction = utils.generate_caption(
            model, 
            tokenizer, 
            input_tensor, 
            max_len=config.MAX_GENERATION_LEN, 
            device=config.DEVICE
        )

        # 1. Exact Match
        if prediction.lower() == true_label.lower() or true_label.lower() in prediction.lower():
            exact_matches += 1
        else:
            if len(errors) < 10:
                errors.append(f"Ground Truth: {true_label} | Prediction: {prediction}")
        
        # 2. Fault Type Match
        true_type = " ".join(true_label.split()[:2]).lower()
        if true_type in prediction.lower(): 
            fault_type_matches += 1

        # 3. Fault Size Match
        for part in true_label.split():
            if "0." in part and part in prediction:
                fault_size_matches += 1
                break

    # Final Report
    accuracy_exact = (exact_matches / total_samples) * 100
    accuracy_type = (fault_type_matches / total_samples) * 100
    accuracy_size = (fault_size_matches / total_samples) * 100

    print("\n" + "="*50)
    print(f"FINAL PERFORMANCE REPORT (Dataset Size: {total_samples})")
    print("="*50)
    print(f"Exact Match Accuracy      : {accuracy_exact:.2f}%")
    print(f"Fault Type Detection      : {accuracy_type:.2f}%")
    print(f"Fault Size Detection      : {accuracy_size:.2f}%")
    print("="*50)

    print("\nError Examples:")
    for err in errors:
        print(f"X {err}")

if __name__ == "__main__":
    evaluate()