FROM nvidia/cuda:10.2-base-centos8

RUN yum install -y https://dl.fedoraproject.org/pub/epel/epel-release-latest-8.noarch.rpm
RUN yum install -y dnf-plugins-core
RUN dnf config-manager --set-enabled PowerTools

RUN yum install -y \
        cuda-cufft-$CUDA_PKG_VERSION \
        fftw-libs \
        hdf5 \
        python3 \
        python3-gdal \
        python3-h5py \
        python3-pip \
        python3-ruamel-yaml \
 && rm -rf /var/cache/yum/*

RUN pip3 install yamale
