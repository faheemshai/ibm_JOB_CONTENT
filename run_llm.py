from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import login
import torch
import sys
import os

# Authenticate with Hugging Face
hf_token = os.getenv("HUGGINGFACE_TOKEN")
if not hf_token:
    print("HUGGINGFACE_TOKEN environment variable not set.")
    exit(1)
login(token=hf_token)

# Model setup
model_id = "mistralai/Mistral-7B-Instruct-v0.2"
quant_config = BitsAndBytesConfig(load_in_4bit=True,
                                  bnb_4bit_quant_type='nf4',
                                  bnb_4bit_use_double_quant=True)

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(model_id,
                                             device_map="auto",
                                             quantization_config=quant_config)

# Get prompt from command-line
if len(sys.argv) < 2:
    print("Usage: run_llm.py <prompt>")
    sys.exit(1)

prompt = sys.argv[1]
if not prompt.startswith("<s>[INST]"):
    prompt = f"<s>[INST] {prompt} [/INST]"

# Tokenize and move to GPU
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Generate output with better settings
output = model.generate(**inputs,
                        max_new_tokens=350,             # Increased token limit
                        temperature=0.7,                # Balanced creativity
                        do_sample=True,                 # Enable sampling to use temperature
                        eos_token_id=tokenizer.eos_token_id)  # Optional, to stop on EOS

# Decode only the newly generated tokens
generated_text = tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Display output
print("\n===== LLM OUTPUT START =====")
print(generated_text.strip())
print("===== LLM OUTPUT END =====\n")
