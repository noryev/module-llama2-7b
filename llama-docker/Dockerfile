# Use PyTorch as base image
FROM pytorch/pytorch:latest

# Set the working directory in the container
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install transformers \
    accelerate \
    bitsandbytes \
    sentencepiece

# Create necessary directories and set permissions
RUN mkdir -p /root/.cache/huggingface /outputs \
    && chmod 777 /outputs

# Download and cache the model (commented out as it requires HF_TOKEN)
# Note: This step should be done at runtime with proper authentication
# RUN --mount=type=secret,id=hf_token \
#     export HF_TOKEN=$(cat /run/secrets/hf_token) && \
#     python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
#     AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=os.environ['HF_TOKEN']); \
#     AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_auth_token=os.environ['HF_TOKEN'])"

# Copy the Python script into the container
COPY run_llama.py /workspace/run_llama.py
RUN chmod +x /workspace/run_llama.py

# Set default environment variables
ENV DEFAULT_PROMPT="Tell me about LLaMA."

# Set the entrypoint to run the Python script and allow for command-line arguments
ENTRYPOINT ["python", "/workspace/run_llama.py"]

# Set a default command that can be overridden
CMD ["${DEFAULT_PROMPT}"]