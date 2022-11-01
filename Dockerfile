FROM ubuntu:20.04

LABEL maintainer="chirag.sakhuja@utexas.edu"

RUN apt-get update \
        && apt-get install -y build-essential wget gcc-9 git libboost-all-dev \
        && wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh \
        && /bin/bash ~/anaconda.sh -b -p /opt/conda \
        && rm ~/anaconda.sh \
        && git clone https://github.com/chiragsakhuja/spotlight.git \
        && git checkout tags/hpca-2023
        && conda env create -f environment.yml

ENV PATH /opt/conda/bin:${PATH}
