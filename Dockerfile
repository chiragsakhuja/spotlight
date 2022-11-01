FROM ubuntu:20.04

LABEL maintainer="chirag.sakhuja@utexas.edu"

RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC apt-get -y install tzdata
RUN apt-get install -y build-essential wget gcc-9 libboost-all-dev \
        && wget --quiet https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh -O ~/anaconda.sh \
        && /bin/bash ~/anaconda.sh -b -p /opt/conda \
        && rm ~/anaconda.sh

ENV APP_PATH=/home
WORKDIR $APP_PATH
COPY . .
ENV PATH /opt/conda/bin:${PATH}
RUN conda env create -f environment.yml
RUN conda init bash
