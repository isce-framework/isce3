## -*- docker-image-name: "isce3:hirsute-dev" -*-

# based on ubuntu
FROM ubuntu:hirsute

# set up some build variables
ARG img=hirsute/dev
ARG imghome=.mm/docker/${img}
# locations
ARG prefix=/usr/local
ARG srcdir=${prefix}/src

# environment
# colorize (for fun)
ENV TERM=xterm-256color
# set up the dynamic linker path
ENV LD_LIBRARY_PATH=${prefix}/lib

# update the package repository
RUN apt update -y
# get the latest
RUN apt dist-upgrade -y

# install the base software stack
RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y \
    git vim unzip zip \
    openssh-server \
    g++ gfortran make cmake googletest \
    libeigen3-dev \
    libfftw3-dev libfftw3-double3 libfftw3-single3 \
    libhdf5-openmpi-dev \
    libgdal-dev \
    libgsl-dev \
    libgtest-dev \
    libpq-dev \
    npm \
    nvidia-cuda-toolkit \
    python3 python3-dev \
    python3-distutils python3-gdal python3-h5py python3-numpy python3-pybind11 python3-pip \
    python3-pytest python3-ruamel.yaml python3-scipy python3-shapely python3-yaml

# no package for yamale
RUN pip3 install yamale

# set up the interactive environment
# go home
WORKDIR /root
# copy the keybindings setup
COPY ${imghome}/inputrc .inputrc
# copy and prep the shell startup file
COPY ${imghome}/bashrc bashrc.in
RUN sed \
    -e "s:@SRCDIR@:${srcdir}:g" \
    bashrc.in > .bashrc
# copy and prep the prompt
COPY ${imghome}/prompt.py prompt.py.in
RUN sed \
    -e "s:@INSTANCE@:${img}:g" \
    prompt.py.in > prompt.py
# the pyre configuration directory
WORKDIR /root/.pyre
# copy and prep the {mm} coniguration file
COPY ${imghome}/mm.yaml mm.yaml.in
RUN sed \
    -e "s:@PREFIX@:${prefix}:g" \
    mm.yaml.in > mm.yaml
# configure mm
WORKDIR /root/.mm
# copy and prep the configuration file
COPY ${imghome}/config.mm config.mm

# go to the source directory
WORKDIR ${srcdir}

# end of file
