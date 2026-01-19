import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# 1. Load Model and Tokenizer
# ---------------------------------------------------------
print("Downloading GPT-2 Model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

print("GPT-2 Ready.")

# GPT-2 does not have a pad token, assign eos (end of sentence)
tokenizer.pad_token = tokenizer.eos_token

# 2. Test: Start a Sentence
# ---------------------------------------------------------
# Test simple text generation.
text = input("Input text: ")
print(f"Input: '{text}'")

# Convert text to number (Tokenization)
inputs = tokenizer(text, return_tensors="pt")

# 3. Text Generation
# ---------------------------------------------------------
print("Generating text...")

with torch.no_grad():
    output_tokens = model.generate(
        **inputs, 
        max_length=30,          # Write max 30 words
        num_return_sequences=1, # Generate 1 response
        do_sample=True,         # Be creative (add randomness)
        temperature=0.7         # Sampling temperature
    )

# 4. Decode Output
# ---------------------------------------------------------
generated_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

print("="*40)
print("GPT-2 Response:")
print(f"--> {generated_text}")
print("="*40)

print("Note: Basic text generation test complete.")