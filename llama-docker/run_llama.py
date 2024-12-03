from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from pathlib import Path
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_prompt():
    # Try command line arguments first
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])  # Join all arguments to handle spaces
    else:
        # Try environment variables next
        prompt = os.environ.get('PROMPT', '')
        # If PROMPT starts with "PROMPT=", remove that prefix
        if prompt.startswith('PROMPT='):
            prompt = prompt[7:]
    
    if not prompt:
        prompt = os.environ.get('DEFAULT_PROMPT', 'What is the capital of France?')
    
    logging.info(f"Using prompt: {prompt}")
    return prompt

def load_model():
    model_name = "meta-llama/Llama-2-7b-hf"
    cache_dir = "/root/.cache/huggingface"
    hf_token = os.environ.get('HF_TOKEN')
    
    if not hf_token:
        raise ValueError("No HF_TOKEN found in environment variables")
    
    try:
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )
        logging.info("Model loaded successfully")
        
        logging.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_auth_token=hf_token,
            cache_dir=cache_dir
        )
        logging.info("Tokenizer loaded successfully")
        
        return model, tokenizer
    except Exception as e:
        raise RuntimeError(f"Model loading failed: {str(e)}")

def generate_text(prompt, model, tokenizer, max_length=100):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    logging.info("Generating response...")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    try:
        # Get prompt using the flexible function
        prompt = get_prompt()
        
        # Load model and generate response
        model, tokenizer = load_model()
        response = generate_text(prompt, model, tokenizer)
        
        # Prepare output
        output = {
            "prompt": prompt,
            "response": response
        }
        
        # Write to output directory
        output_dir = Path('/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'result.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        # Also print to stdout
        print("\nResponse:", response)
        
    except Exception as e:
        logging.error(f"Error: {str(e)}")
        error_output = {
            "error": str(e),
            "troubleshooting": [
                "Verify HF_TOKEN is correct",
                "Check model access permissions",
                "Ensure sufficient GPU memory"
            ]
        }
        
        output_dir = Path('/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / 'error.json', 'w') as f:
            json.dump(error_output, f, indent=2)
        sys.exit(1)