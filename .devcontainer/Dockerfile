FROM osrf/ros:humble-desktop-full

# Add a non-root user
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME

# Install common dev tools
RUN apt-get update && apt-get install -y \
    python3-pip \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge
ENV CONDA_DIR=/opt/conda
RUN mkdir -p $CONDA_DIR && chown $USERNAME:$USERNAME $CONDA_DIR

USER $USERNAME
WORKDIR /home/$USERNAME

RUN wget --quiet https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-$(uname -m).sh -O ~/miniforge.sh \
    && /bin/bash ~/miniforge.sh -b -p $CONDA_DIR \
    && rm ~/miniforge.sh

# Configure environment
ENV PATH=$CONDA_DIR/bin:$PATH

# Add mamba_on/mamba_off to .bashrc and prevent auto-activation of base
RUN echo 'mamba_on() {\n    source "/opt/conda/etc/profile.d/conda.sh"\n    source "/opt/conda/etc/profile.d/mamba.sh"\n    conda activate base\n}' >> /home/$USERNAME/.bashrc \
    && echo 'mamba_off() {\n    conda deactivate\n}' >> /home/$USERNAME/.bashrc \
    && conda config --set auto_activate_base false \
    && conda clean --all --yes

ENV SHELL /bin/bash
