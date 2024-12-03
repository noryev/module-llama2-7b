from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import json
from pathlib import Path

# Get prompt and clean it
prompt = os.environ.get('PROMPT', '')
# Remove any quotes
prompt = prompt.strip('"')

# Print for debugging
print(f"Received prompt: {prompt}")

try:
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_auth_token=os.environ.get('HF_TOKEN')
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf", 
        use_auth_token=os.environ.get('HF_TOKEN')
    )

    # Generate response
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        do_sample=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Save output
    output = {
        "prompt": prompt,
        "response": response
    }

    # Write to outputs directory
    output_dir = Path('/outputs')
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / 'result.json', 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResponse: {response}")

except Exception as e:
    print(f"Error: {str(e)}")
    with open('/outputs/error.json', 'w') as f:
        json.dump({"error": str(e)}, f)