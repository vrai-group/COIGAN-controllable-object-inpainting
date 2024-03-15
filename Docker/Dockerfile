FROM nvidia/cuda:11.6.2-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.2-devel-ubuntu20.04
#FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

RUN apt-get update 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.8 python3-pip

# Update symlink to point to latest
RUN rm /usr/bin/python3 && ln -s /usr/bin/python3.8 /usr/bin/python3
RUN python3 --version
RUN pip3 --version

# System and basic libraries
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    libqt5core5a \
    libqt5gui5 \
    libqt5widgets5 \
    libqt5dbus5 \
    qttools5-dev \
    qttools5-dev-tools \
    libx11-dev \
    libxext-dev \
    libxrender-dev \
    libfontconfig1-dev \
    libfreetype6-dev \
    libxfixes-dev \
    libx11-xcb-dev \
    libxcb-glx0-dev \
	git \
    unzip \
    wget \
    tmux

#RUN python3 --version
#RUN pip3 --version
RUN pip install torch==1.12.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
RUN pip install	torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116

# install GitHub CLI
RUN type -p curl >/dev/null || (apt update && apt install curl -y)
RUN curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
RUN chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
RUN echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli.list > /dev/null
RUN apt update
RUN apt install gh -y

#enviroment definiscion
ENV APPDIR coigan
WORKDIR coigan

# just a trigger to download the repo again without using the cache
RUN echo "modify this string to download the new version of the repo 0003"

#DEBUG ENV VARIABLES
ENV HYDRA_FULL_ERROR=1

RUN cd /coigan/
RUN git clone https://github.com/MassimilianoBiancucci/COIGAN-IROS-2024.git
RUN pip install -e /coigan/COIGAN-IROS-2024/COIGAN/

# Ninja needed for FCC model fused layers (tested ninja 1.10.1)
RUN pip install ninja

# Set kaggle credentials
# !!! Remeber to create the build_credentials file as described in the README in "Setup the build_args file"
ARG KAGGLE_USERNAME
ARG KAGGLE_KEY
ENV KAGGLE_USERNAME=${KAGGLE_USERNAME}
ENV KAGGLE_KEY=${KAGGLE_KEY}

# set wandb module and credentials
# !!! Remeber to create the build_credentials file as described in the README in "Setup the build_args file"
RUN pip install wandb
ARG WANDB_API_KEY
RUN wandb login ${WANDB_API_KEY}

# Download and prepare the severstal dataset
# RUN python3 /coigan/COIGAN-IROS-2024/scripts/prepare_severstal_dataset.py

# Downloading the models used for the loss
RUN python3 /coigan/COIGAN-IROS-2024/scripts/download_models_loss.py

# Setting up the jupyter notebook
RUN pip install jupyterlab

RUN mkdir /notebooks
#CMD jupyter notebook --no-browser --ip 0.0.0.0 --port 8888  --allow-root /notebooks

#ENTRYPOINT [ "tail", "-f", "/dev/null" ]
CMD ["bash"]