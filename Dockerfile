FROM nvidia/cuda:12.3.0-runtime-ubuntu22.04

# Install Python and pip
RUN apt-get update && \
    apt-get install -y git python3-pip && \
    pip3 install --upgrade pip

# Install required Python packages
RUN pip3 install torch==2.3.0+cu121 --extra-index-url https://download.pytorch.org/whl/cu121 && \
    pip3 install transformers==4.42.0 accelerate==0.28.0 sentencepiece bitsandbytes==0.43.1

# Set working directory
WORKDIR /app

# Copy files
COPY run_llm.py .
COPY entrypoint.sh .

# Make the entrypoint script executable
RUN chmod +x entrypoint.sh

# Run entrypoint script
ENTRYPOINT ["./entrypoint.sh"]
