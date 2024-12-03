FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece python-dotenv huggingface_hub

# Create necessary directories
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /outputs

# Copy the Python script
COPY run_llama.py /app/

# Make the script write to the outputs directory
ENV OUTPUT_DIR=/outputs

ENTRYPOINT ["python", "run_llama.py"]