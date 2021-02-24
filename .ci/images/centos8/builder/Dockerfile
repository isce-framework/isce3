FROM isce-ci-centos8/base

RUN yum install -y \
        cuda-cudart-dev-10-2 \
        cuda-cufft-dev-10-2 \
        cuda-nvcc-10-2 \
        git \
 && rm -rf /var/cache/yum/*

RUN conda install -y \
        cmake \
        fftw \
        gdal \
        git \
        h5py \
        hdf5 \
        make \
        numpy \
 && rm -rf /usr/local/conda/pkgs

# set up permissions
ARG UNAME=user
ARG UID=1000
ARG GID=1000

ARG SRCDIR
ARG BLDDIR
ARG PREFIX

# create build/install volumes
# https://github.com/moby/moby/issues/5419#issuecomment-41478290
RUN groupadd -g $GID -o $UNAME \
 && useradd --no-log-init -m -u $UID -g $GID -o -s /bin/bash $UNAME \
 && mkdir $BLDDIR && chown $UID:$GID $BLDDIR \
 && mkdir $PREFIX && chown $UID:$GID $PREFIX

VOLUME $BLDDIR $PREFIX

USER $UNAME
WORKDIR $BLDDIR
