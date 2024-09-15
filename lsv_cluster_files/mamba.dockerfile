# For LSV A100s server
FROM nvcr.io/nvidia/pytorch:22.02-py3

# Set path to CUDA
ENV CUDA_HOME=/usr/local/cuda

# Install additional programs
RUN apt-get update && apt-get install -y\
    build-essential \
    git \
    htop \
    gnupg \
    curl \
    ca-certificates \
    vim \
    tmux \
    && rm -rf /var/lib/apt/lists/*

# Explicitly install Python packages and check CUDA
#RUN python -m pip install --upgrade pip "setuptools<71"
#RUN python -m pip install torch

RUN python3 -m pip install --upgrade pip "setuptools==69.5.1"

RUN python3 -m pip install torch

RUN git clone https://github.com/rozariwang/causal-conv1d.git /opt/causal-conv1d
RUN cd /opt/causal-conv1d && pip install -v . --no-cache-dir --no-build-isolation 

# Install Python dependencies
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
    # \
    #causal-conv1d

#RUN python3 -m pip install -v mamba-ssm

# Clone the mamba repository
RUN git clone https://github.com/state-spaces/mamba.git /opt/mamba
#RUN git clone https://github.com/rozariwang/mamba.git /opt/mamba
# Install mamba from the cloned repository
RUN cd /opt/mamba && pip install -v . --no-cache-dir --no-build-isolation

#RUN python3 -m pip install mamba-ssm --no-cache-dir --no-build-isolation
# Specify a new user (USER_NAME and USER_UID are specified via --build-arg)
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