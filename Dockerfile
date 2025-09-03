FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Avoid timezone prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    libgomp1 \
    git \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p trained_models results analysis_results

# Set environment variables
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/.torch

CMD ["python", "setup_and_train.py"]