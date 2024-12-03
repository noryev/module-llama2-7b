from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_inference():
    # Get prompt from environment variable
    prompt = os.environ.get('PROMPT', '')
    if not prompt:
        raise ValueError("No PROMPT found in environment variables")
    
    logging.info(f"Processing prompt: {prompt}")
    
    # Get HF token
    hf_token = os.environ.get('HF_TOKEN')
    if not hf_token:
        raise ValueError("No HF_TOKEN found in environment variables")
    
    try:
        # Load model
        logging.info("Loading model...")
        model = AutoModelForCausalLM.from_pretrained(
            "meta-llama/Llama-2-7b-hf",
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            use_auth_token=hf_token,
            cache_dir="/root/.cache/huggingface"
        )
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-hf", 
            use_auth_token=hf_token,
            cache_dir="/root/.cache/huggingface"
        )
        
        # Generate response
        logging.info("Generating response...")
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        outputs = model.generate(
            **inputs,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True
        )
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Write output
        output = {
            "prompt": prompt,
            "response": response
        }
        
        # Save to outputs directory
        output_dir = Path('/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'result.json', 'w') as f:
            json.dump(output, f, indent=2)
            
        logging.info("Generation completed successfully")
        
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        error_output = {
            "error": str(e),
            "prompt": prompt
        }
        
        output_dir = Path('/outputs')
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / 'error.json', 'w') as f:
            json.dump(error_output, f, indent=2)
        raise

if __name__ == "__main__":
    run_inference()