import torch
import torch.nn as nn
from transformers import ASTModel, GPT2LMHeadModel
import config

class SignalCaptioningModel(nn.Module):
    def __init__(self, encoder_name=config.ENCODER_ID, decoder_name=config.DECODER_ID):
        super(SignalCaptioningModel, self).__init__()

        print(f"Loading Encoder: {encoder_name}...")
        self.encoder = ASTModel.from_pretrained(encoder_name)

        print(f"Loading Decoder: {decoder_name}...")
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name)

        self.encoder_dim = config.ENCODER_DIM
        self.decoder_dim = config.DECODER_DIM

        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )

    def forward(self, input_values, labels=None):
        """
        Forward pass for the model.
        Args:
            input_values: Processed audio features from ASTFeatureExtractor.
            labels: usage for training, tokenized target sentences.
        """
        
        # 1. Encode the audio signal
        encoder_outputs = self.encoder(input_values)

        # 2. Project encoder output to decoder dimension
        signal_embeds = self.projection(encoder_outputs.pooler_output).unsqueeze(1)

        if labels is not None:
            # --- Training Phase ---
            
            # Get embeddings for the target text
            word_embeds = self.decoder.transformer.wte(labels)

            # Concatenate signal embeddings with text embeddings
            full_embeds = torch.cat((signal_embeds, word_embeds), dim=1)

            # Create labels for loss calculation (ignore index -100 for the signal part)
            ignore_index = torch.full((labels.size(0), 1), -100, device=labels.device)
            full_labels = torch.cat((ignore_index, labels), dim=1)

            # Forward pass through decoder
            outputs = self.decoder(inputs_embeds=full_embeds, labels=full_labels)
            return outputs
        
        else:
            # --- Inference Phase ---
            return signal_embeds