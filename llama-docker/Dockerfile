FROM pytorch/pytorch:latest

WORKDIR /app

# Install required packages
RUN pip install transformers accelerate bitsandbytes sentencepiece python-dotenv huggingface_hub

# Create necessary directories
RUN mkdir -p /root/.cache/huggingface
RUN mkdir -p /outputs \
    && chmod 777 /outputs

# Copy the Python script
COPY run_llama.py /app/
RUN chmod +x /app/run_llama.py

# Set default environment variables
ENV DEFAULT_PROMPT="What is the capital of France?"

# Set the entrypoint
ENTRYPOINT ["python", "/app/run_llama.py"]

# Allow for command-line arguments
CMD ["${DEFAULT_PROMPT}"]