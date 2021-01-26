FROM alpine

RUN apk add --no-cache \
        fftw \
        gdal \
        hdf5 \
        libgomp \
        libstdc++ \
        python3 \
        py3-gdal \
        py3-pip \
        py3-ruamel.yaml \
 && pip install yamale \
 && apk del py3-pip \
 && echo http://dl-cdn.alpinelinux.org/alpine/edge/testing \
        >> /etc/apk/repositories \
 && apk add --no-cache py3-h5py
