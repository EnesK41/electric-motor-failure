import torch
import random
import numpy as np

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_caption(model, tokenizer, input_tensor, max_len=20, device='cpu'):
    """
    Generates a caption for the given input tensor using the model.
    Implements a repetition penalty to prevent loops in generation.
    """
    model.eval()
    with torch.no_grad():
        # Encode signal
        encoder_outputs = model.encoder(input_tensor)
        signal_embeds = model.projection(encoder_outputs.pooler_output).unsqueeze(1)
        
        current_input = signal_embeds
        generated_ids = []
        
        for _ in range(max_len):
            outputs = model.decoder(inputs_embeds=current_input)
            next_token_logits = outputs.logits[:, -1, :] 
            
            # Repetition Penalty
            # If the logit is negative, multiplying by a positive number makes it smaller (more negative, less likely).
            # If the logit is positive, dividing by a positive number makes it smaller (closer to zero, less likely).
            for token_id in generated_ids:
                if next_token_logits[0, token_id] < 0:
                    next_token_logits[0, token_id] *= 5.0
                else:    
                    next_token_logits[0, token_id] /= 5.0
            
            # Select the most likely token
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
            token_val = next_token_id.item()
            
            # Stop if EOS token is generated
            if token_val == tokenizer.eos_token_id:
                break
            
            generated_ids.append(token_val)
            
            # Update input for next step
            new_word_embed = model.decoder.transformer.wte(next_token_id)
            current_input = torch.cat((current_input, new_word_embed), dim=1)

        caption = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        return caption
