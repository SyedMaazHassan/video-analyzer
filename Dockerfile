FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Noninteractive apt, stable matplotlib backend for headless envs
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MPLBACKEND=Agg \
    PIP_NO_CACHE_DIR=1

# System deps:
# - build-essential: needed for C/C++ ext builds (e.g., line-profiler)
# - tzdata: to satisfy TZ
# - ffmpeg: for video IO
# - libgomp1: OpenMP (scikit-learn, etc.)
# - libglib2.0-0, libsm6, libxext6, libxrender1: common image/GUI libs some wheels expect
# - git: if pip needs to install from git+...
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tzdata \
    ffmpeg \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    git \
 && rm -rf /var/lib/apt/lists/*

# (Optional) If you truly need Tk at runtime, uncomment:
# RUN apt-get update && apt-get install -y --no-install-recommends python3-tk && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip toolchain before installing your deps
RUN python -m pip install -U pip setuptools wheel

# Copy only the requirements first to leverage cache
COPY requirements.txt /app/requirements.txt

# Install Python deps (ensure your requirements are cleaned as discussed)
RUN pip install --no-compile -r /app/requirements.txt

# Copy project
COPY . .

# Ensure expected directories exist
RUN mkdir -p trained_models results analysis_results

# Helpful envs
ENV PYTHONPATH=/app
ENV TORCH_HOME=/app/.torch

# Default command - run training
CMD ["python", "surgical_ai_system/training/practical_master_trainer.py"]
