import torch
from transformers import ASTFeatureExtractor
from dataset import CWRUDataset
from model import SignalCaptioningModel

# 1. SETTINGS AND PREPARATION
data_path = 'data/raw'
encoder_id = "MIT/ast-finetuned-audioset-10-10-0.4593"
decoder_id = "gpt2"

print("Setting up test environment...")

try:
    # Feature Extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(encoder_id)

    dataset = CWRUDataset(data_path, feature_extractor)
    
    model = SignalCaptioningModel(encoder_id, decoder_id)
    model.eval()

    # 2. Test Inference
    # ---------------------------------------------------
    if len(dataset) == 0:
        print("Error: 'data/raw' folder is empty. Download .mat files first.")
    else:
        # Get first file
        sample_input, sample_label = dataset[0]
        
        print(f"\nTested File Label: {sample_label}")
        print(f"Signal Size (AST Input): {sample_input.shape}")

        # Run model once (Forward pass only)
        # Reshape signal to [1, 1024, 128] (Add batch dimension)
        with torch.no_grad():
            input_tensor = sample_input.unsqueeze(0)
            
            # For now, manually test encoder and projection parts:
            
            # Pass through Encoder
            enc_out = model.encoder(input_tensor).pooler_output
            # Pass through Bridge/Projection
            proj_out = model.projection(enc_out)

            print(f"Projected Vector Size: {proj_out.shape}")
            print("\nPipeline execution successful.")
            print("Data flow verified: Disk -> Preprocessing -> Model -> Projection.")

except Exception as e:
    print(f"\nError occurred during test:\n{e}")