import torch
import torch.nn as nn
import os
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, GPT2Tokenizer
from dataset import CWRUDataset
from model import SignalCaptioningModel
import config

def train():
    # Setup Device
    print(f"Device: {config.DEVICE}")

    # Initialize Tokenizer & Feature Extractor
    feature_extractor = ASTFeatureExtractor.from_pretrained(config.ENCODER_ID)
    tokenizer = GPT2Tokenizer.from_pretrained(config.DECODER_ID)
    
    # Ensure pad_token is set (GPT-2 does not have one by default)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare Dataset
    print("Preparing dataset...")
    dataset = CWRUDataset(config.RAW_DATA_DIR, feature_extractor)
    train_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    # Initialize Model
    model = SignalCaptioningModel(config.ENCODER_ID, config.DECODER_ID).to(config.DEVICE)

    # Load Existing Model if Available (Resume Training)
    if os.path.exists(config.MODEL_SAVE_PATH):
        print(f"Loading existing model from {config.MODEL_SAVE_PATH}")
        try:
            model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
            print("Model loaded successfully. Resuming training...")
        except Exception as e:
            print(f"Warning: Could not load model. Starting from scratch. Error: {e}")
    else:
        print("No existing model found. Starting fresh training.")

    # Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2, verbose=True
    )

    # Training Variables
    best_loss = float('inf')
    patience_counter = 0

    model.train()
    print(f"\nStarting training for {config.EPOCHS} epochs...")

    for epoch in range(config.EPOCHS):
        total_loss = 0
        print(f"--- Epoch {epoch+1}/{config.EPOCHS} (LR: {optimizer.param_groups[0]['lr']:.6f}) ---")
        
        for batch_idx, (signals, labels) in enumerate(train_loader):
            signals = signals.to(config.DEVICE)
            
            # Tokenize targets
            targets = tokenizer(
                labels, 
                return_tensors="pt", 
                padding="max_length", 
                max_length=config.MAX_GENERATION_LEN, 
                truncation=True
            ).input_ids.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(signals, labels=targets)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 20 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

        # End of Epoch
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Completed. Average Loss: {avg_loss:.4f}")

        # Update Scheduler
        scheduler.step(avg_loss)

        # Save Best Model
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved! Loss: {best_loss:.4f}")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config.PATIENCE}")
            
            if patience_counter >= config.PATIENCE:
                print(f"\nEarly Stopping triggered.")
                break

    print("\nTraining Completed.")

if __name__ == "__main__":
    train()