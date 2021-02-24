FROM nvidia/cuda:10.1-devel-ubuntu19.04

# get prereqs
RUN apt-get update && apt-get install -y \
        cmake \
        fftw3-dev \
        git \
        libeigen3-dev \
        libgdal-dev \
        libgtest-dev \
        libhdf5-dev \
        locales \
        python3-dev \
        python3-numpy \
 && rm -rf /var/lib/apt/lists/*

# set up locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8

# set up permissions
ARG UNAME=user
ARG UID=1000
ARG GID=1000

ARG SRCDIR
ARG BLDDIR
ARG PREFIX

# create build/install volumes
RUN groupadd -g $GID -o $UNAME \
 && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME \
 && mkdir $BLDDIR && chown $UID:$GID $BLDDIR \
 && mkdir $PREFIX && chown $UID:$GID $PREFIX
VOLUME $BLDDIR $PREFIX

USER $UNAME
WORKDIR $BLDDIR
