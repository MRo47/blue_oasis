FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ARG USERNAME=ubuntu
ARG VENV_PATH=/home/${USERNAME}/.venv

# dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl unzip ffmpeg sox libsndfile1 \
    build-essential pkg-config \
    libavformat-dev libavcodec-dev libavutil-dev \
    libsndfile1-dev libffi-dev libssl-dev \
    sudo libpng-dev libfreetype6-dev libopenblas-dev git-lfs \
    && rm -rf /var/lib/apt/lists/*

# make sure non root can use sudo
RUN echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

USER ${USERNAME}

RUN python3 -m venv ${VENV_PATH} \
    && echo "source ${VENV_PATH}/bin/activate" >> /home/${USERNAME}/.bashrc

ENV PATH="${VENV_PATH}/bin:$PATH"

# Copy and install dependencies from requirements.txt
COPY bird_classifier/requirements.txt /tmp/requirements.txt

RUN pip install --upgrade pip setuptools wheel

RUN pip install --no-cache-dir -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["/bin/bash"]