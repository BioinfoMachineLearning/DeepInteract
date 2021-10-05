ARG CUDA_FULL=11.2.2
FROM nvidia/cuda:${CUDA_FULL}-cudnn8-runtime-ubuntu20.04
# FROM directive resets ARGS, so we specify again (the value is retained if
# previously set).
ARG CUDA_FULL
ARG CUDA=11.2

# Use bash to support string substitution.
SHELL ["/bin/bash", "-c"]

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
      build-essential \
      cmake \
      cuda-command-line-tools-${CUDA/./-} \
      git \
      wget \
      software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Compile PSAIA from source.
# First, install GCC 10 for Ubuntu 20.04.
RUN add-apt-repository ppa:ubuntu-toolchain-r/ppa \
    && apt-get update \
    && apt-get install -y gcc-10 g++-10
# Second, install QT4 for Ubuntu 20.04.
RUN add-apt-repository ppa:rock-core/qt4 \
    && apt-get update \
    && apt-get install -y libqt4* libqtcore4 libqtgui4 libqtwebkit4 qt4* libxext-dev
# Then, begin compiling PSAIA along with PSA and PIA.
RUN mkdir -p /home/Programs
WORKDIR /home/Programs
RUN wget http://complex.zesoi.fer.hr/data/PSAIA-1.0-source.tar.gz \
    && tar -xvzf PSAIA-1.0-source.tar.gz
WORKDIR PSAIA_1.0_source/make/linux/psaia/
RUN qmake-qt4 psaia.pro \
    && make
WORKDIR ../psa/
RUN qmake-qt4 psa.pro \
    && make
WORKDIR ../pia/
RUN qmake-qt4 pia.pro \
    && make

# Compile HHsuite from source.
RUN git clone --branch v3.3.0 https://github.com/soedinglab/hh-suite.git /tmp/hh-suite \
    && mkdir /tmp/hh-suite/build
WORKDIR /tmp/hh-suite/build
RUN cmake -DCMAKE_INSTALL_PREFIX=/opt/hhsuite .. \
    && make -j 4 && make install \
    && ln -s /opt/hhsuite/bin/* /usr/bin \
    && rm -rf /tmp/hh-suite

# Install Miniconda package manager.
RUN wget -q -P /tmp \
  https://repo.anaconda.com/miniconda/Miniconda3-py38_4.9.2-Linux-x86_64.sh \
    && bash /tmp/Miniconda3-py38_4.9.2-Linux-x86_64.sh -b -p /opt/conda \
    && rm /tmp/Miniconda3-py38_4.9.2-Linux-x86_64.sh

# Install conda packages.
ENV PATH="/opt/conda/bin:$PATH"
RUN conda update -qy conda \
    && conda install -y -c pytorch -c conda-forge -c defaults -c anaconda -c bioconda -c salilab \
      python==3.8 \
      pip==21.1.2 \
      cudatoolkit==${CUDA} \
      pytorch==1.7.1 \
      torchvision==0.8.2 \
      torchaudio==0.7.2 \
      numpy==1.21.2 \
      scipy==1.4.1 \
      pandas==1.2.4 \
      scikit-learn==0.24.2 \
      requests==2.26.0 \
      biopython==1.78 \
      hhsuite==3.3.0 \
      msms==2.6.1 \
      dssp==3.0.0 \
      aria2==1.34.0

# Mirror the curated directory structure in the Docker image's application execution directory.
COPY . /app/DeepInteract

# Install pip packages.
WORKDIR /app/DeepInteract
RUN pip3 install --upgrade pip \
    && pip3 install -e . \
    && pip3 install -r /app/DeepInteract/requirements.txt \
    && pip3 install https://data.dgl.ai/wheels/dgl_cu110-0.6-cp38-cp38-manylinux1_x86_64.whl

# Remove Git artifacts from local repository clone.
RUN rm -rf .git/

# We need to run `ldconfig` first to ensure GPUs are visible, due to some quirk
# with Debian. See https://github.com/NVIDIA/nvidia-docker/issues/1399 for details.
# ENTRYPOINT does not support easily running multiple commands, so instead we
# write a shell script to wrap them up.
RUN echo $'#!/bin/bash\n\
ldconfig\n\
python /app/DeepInteract/project/lit_model_predict_docker.py "$@"' > /app/run_deepinteract.sh \
  && chmod +x /app/run_deepinteract.sh
ENTRYPOINT ["/app/run_deepinteract.sh"]