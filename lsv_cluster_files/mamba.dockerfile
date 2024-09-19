# For LSV A100s server
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

ENV DEBIAN_FRONTEND=noninteractive

# Add deadsnakes PPA for newer Python versions
RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa

# Install additional programs including Python 3.10
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    python3.10 \
    python3.10-dev \
    python3.10-venv \
    python3-pip \
    python3-wheel \
    cuda-command-line-tools-12-1 \
    cuda-cudart-dev-12-1 \
    cuda-cudart-12-1 \
    nvidia-utils-450 \
    && rm -rf /var/lib/apt/lists/*

RUN nvcc --version || echo "nvcc not found"
RUN ldconfig -p | grep cuda || echo "CUDA libraries not found"
RUN nvidia-smi || echo "nvidia-smi not found"

# Upgrade pip and install setuptools and wheel
RUN pip3 install --upgrade pip setuptools wheel

RUN python3 -m pip install --upgrade pip "setuptools==69.5.1"

# Install specific versions of PyTorch and CUDA that are compatible with mamba-ssm
RUN pip3 install torch

# Optionally install development dependencies if needed
# RUN pip3 install mamba-ssm[dev] --no-cache-dir --no-build-isolation

#RUN git clone https://github.com/Dao-AILab/causal-conv1d.git /opt/causal-conv1d

RUN python3 -m pip install packaging

#RUN cd /opt/causal-conv1d && pip3 install -v . --no-cache-dir --no-build-isolation 

# Clone and install mamba repository
RUN git clone https://github.com/state-spaces/mamba.git /opt/mamba
RUN cd /opt/mamba && pip3 install . --no-cache-dir --no-build-isolation


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

RUN pip install --no-cache-dir \
    triton==2.2.0 \
    causal_conv1d \
    tqdm \
    torch torchvision torchaudio \
    einops


# Uninstall and Reinstall mamba-ssm with no cache
#RUN pip3 uninstall mamba-ssm -y
#RUN pip3 install mamba-ssm --no-cache-dir
# Install mamba-ssm with its dependencies
RUN pip3 install mamba-ssm --no-build-isolation
RUN pip3 install mamba-ssm[causal-conv1d] --no-cache-dir --no-build-isolation
RUN pip3 install mamba-ssm[dev] --no-build-isolation


# Create a new user with specified USER_UID and USER_NAME
ARG USER_UID
ARG USER_NAME
ENV USER_GID=$USER_UID
ENV USER_GROUP="users"
RUN mkdir /home/$USER_NAME && \
    useradd -l -d /home/$USER_NAME -u $USER_UID -g $USER_GROUP $USER_NAME && \
    mkdir /home/$USER_NAME/.local && \
    chown -R ${USER_UID}:${USER_GID} /home/$USER_NAME/

CMD ["/bin/bash"]
