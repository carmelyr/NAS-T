# Use NVIDIA PyTorch image with CUDA support
FROM nvcr.io/nvidia/pytorch:21.12-py3

# Set the working directory inside the container
WORKDIR /app

# Copy your extended_nas_t directory into the container
COPY extended_nas_t /app/extended_nas_t

# Copy the classification_ozone dataset
COPY classification_ozone /app/classification_ozone

# Install required Python packages
RUN pip install --upgrade pip && \
    pip install -r /app/extended_nas_t/requirements.txt

# Set the default command to run your training script
CMD ["python3", "/app/extended_nas_t/main.py"]

