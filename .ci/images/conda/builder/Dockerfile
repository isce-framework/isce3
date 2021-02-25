FROM isce-ci-conda/base

# get prereqs
RUN conda install -q -y \
        cmake>=3.12 \
        eigen \
        fftw \
        gdal \
        git \
        h5py \
        hdf5 \
        make \
        numpy \
 && conda clean --all --yes

RUN apt-get update && apt-get install -y \
        g++-6 \
        cuda-cudart-dev-10-1 \
        cuda-cufft-dev-10-1 \
        cuda-nvcc-10-1 \
 && rm -rf /var/lib/apt/lists/*

# set up locales
ENV LANG en_US.UTF-8

# set up permissions
ARG UNAME=user
ARG UID=1000
ARG GID=1000

ARG SRCDIR
ARG BLDDIR
ARG PREFIX
ARG DOCDIR

# create build/install volumes
RUN groupadd -g $GID -o $UNAME \
 && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME \
 && mkdir $BLDDIR && chown $UID:$GID $BLDDIR \
 && mkdir $PREFIX && chown $UID:$GID $PREFIX
VOLUME $DOCDIR $BLDDIR $PREFIX

USER $UNAME
WORKDIR $BLDDIR
