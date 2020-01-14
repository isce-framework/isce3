FROM isce-ci-centos8/base

RUN yum install -y \
        cuda-cudart-10-2 \
        cuda-cufft-10-2 \
        git \
 && rm -rf /var/cache/yum/*

RUN dnf --enablerepo=PowerTools install -y cppcheck \
 && rm -rf /var/cache/yum/*

RUN conda install -q -y \
        cmake \
        fftw \
        gdal \
        h5py \
        numpy \
        pytest \
        wget \
 && rm -rf /usr/local/conda/pkgs

ARG SRCDIR
ARG BLDDIR
ARG PREFIX

WORKDIR $BLDDIR

RUN echo ". /usr/local/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate base" >> ~/.bashrc
