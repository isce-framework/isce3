FROM nvidia/cuda:10.1-devel-ubuntu19.04

# runtime libraries
# cmake needed for ctest
RUN apt-get update && apt-get install -y \
        cmake \
        cppcheck \
        lcov \
        libcufft10 \
        libfftw3-3 \
        libgdal20 \
        libgtest-dev \
        libhdf5-103 \
        libhdf5-cpp-103 \
        python3 \
        python3-dev \
        python3-distutils-extra \
        python3-gdal \
        python3-h5py \
        python3-numpy \
        python3-pytest \
        wget \
 && rm -rf /var/lib/apt/lists/*

ARG SRCDIR
ARG BLDDIR
ARG PREFIX

WORKDIR $BLDDIR
