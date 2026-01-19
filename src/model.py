import torch
import torch.nn as nn
from transformers import ASTModel, GPT2LMHeadModel

class SignalCaptioningModel(nn.Module):
    def __init__(self, encoder_name, decoder_name):
        super(SignalCaptioningModel, self).__init__()

        print(f"Loading Encoder: {encoder_name}...")
        self.encoder = ASTModel.from_pretrained(encoder_name)

        print(f"Loading Decoder: {decoder_name}...")
        self.decoder = GPT2LMHeadModel.from_pretrained(decoder_name)


        self.encoder_dim = 768
        self.decoder_dim = 768

        self.projection = nn.Sequential(
            nn.Linear(self.encoder_dim, self.decoder_dim),
            nn.ReLU(),   #It filters noise(Negative signals) and also gives our model the ability to think
            nn.Dropout(0.1) #To prevent memorization, also called Regularization?
        )

    def forward(self, input_values, labels=None):
        # input: processed signals from AST
        # labels: target sentences

        #Signal to encoder
        encoder_outputs = self.encoder(input_values)

        #Translate it
        signal_embedding = encoder_outputs.pooler_output

        projected_embedding = self.projection(signal_embedding).unsqueeze(1)

        if labels is not None:
            #Training

            #Give both the answer and signal to calculate the loss
            outputs =   self.decoder(input_embeds = projected_embedding , labels = labels)
            return outputs
        else:
            #Production
            
            pass

print("Model initialized successfully.")