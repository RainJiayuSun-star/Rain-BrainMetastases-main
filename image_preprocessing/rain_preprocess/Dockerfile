# Premium IDiA Lab Unified Preprocessing Pipeline Dockerfile
# Optimized for clinical neuroimaging operations with PyTorch, CUDA 12.1, and SimpleITK

FROM pytorch/pytorch:2.1.2-cuda12.1-cudnn8-runtime

LABEL maintainer="IDiA Lab"
LABEL version="1.0"
LABEL description="Docker environment for running IDiA Lab Unified Preprocessing Pipeline (rain_preprocess)"

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install essential system dependencies and utilities
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    nano \
    && rm -rf /var/lib/apt/lists/*

# Set standard workspace directory
WORKDIR /workspace

# Upgrade pip and install build tools
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Install medical imaging, pipeline execution, and deep learning dependencies
RUN pip install --no-cache-dir \
    SimpleITK \
    pandas \
    numpy \
    PyYAML \
    hd-bet \
    nnunetv2 \
    tqdm \
    openpyxl \
    nibabel

# Copy local preprocessing pipeline code into the Docker image
# This assumes the build context is set to the 'rain_preprocess' directory
COPY . /workspace/rain_preprocess/

# Create a customized welcoming message for bash shells inside the container
RUN echo 'echo -e "\\e[1;35m============================================================\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;35m    ⚡ IDiA LAB UNIFIED PREPROCESSING PIPELINE (RAIN) ⚡    \\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;35m============================================================\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;32m  * Modalities : Multi-modal MRI (T1-CE, T2-FLAIR, and Masks)\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;32m  * Operations : 1: Co-reg, 2: Skull-Strip, 3: N4, 5: Warp, 6: Normalization\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;32m  * Workspace  : Located at /workspace/rain_preprocess/\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;32m  * Environment: PyTorch 2.1.2 & SimpleITK (CUDA 12.1 Enabled)\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;33m  * Run command: \`cd /workspace/rain_preprocess && python rain_preprocess.py config.yaml\`\\e[0m"' >> /root/.bashrc && \
    echo 'echo -e "\\e[1;35m============================================================\\e[0m"' >> /root/.bashrc

CMD ["/bin/bash"]
