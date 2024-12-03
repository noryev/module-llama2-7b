FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece

# Create necessary directories
RUN mkdir -p /root/.cache/huggingface /outputs

# Copy script
COPY run_llama.py .

ENTRYPOINT ["python", "run_llama.py"]