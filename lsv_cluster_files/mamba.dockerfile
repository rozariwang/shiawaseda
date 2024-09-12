# For LSV A100s server
FROM nvcr.io/nvidia/pytorch:23.02-py3

# Add NVIDIA package repository
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /" > /etc/apt/sources.list.d/cuda.list

# Update and install CUDA 12.x
RUN apt-get update && apt-get install -y cuda-12-0

# Set path to CUDA
ENV PATH=/usr/local/cuda-12.0/bin${PATH:+:${PATH}}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# Install additional programs
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN python3 -m pip install --upgrade pip "setuptools==69.5.1"
RUN python3 -m pip install torch

# Clone and install causal-conv1d
RUN git clone https://github.com/rozariwang/causal-conv1d.git /opt/causal-conv1d
RUN cd /opt/causal-conv1d && pip install .

# Install other Python dependencies
RUN python3 -m pip install \
    accelerate \
    wandb \
    optuna \
    pandas \
    scikit-learn \
    transformers \
    plotly \
    matplotlib \
    rdkit-pypi \
    datasets \
    ninja

# Clone and install mamba
RUN git clone https://github.com/state-spaces/mamba.git /opt/mamba
RUN cd /opt/mamba && pip install .

# Specify a new user
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"

# Create the user
RUN mkdir /home/$USER_NAME && \
    useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME && \
    mkdir /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]