FROM isce-ci-conda/base

# runtime libraries
# cmake needed for ctest
RUN apt-get update && apt-get install -y \
        cppcheck \
        cuda-cudart-10-1 \
        cuda-cufft-10-1 \
        doxygen \
        lcov \
        libfftw3-3 \
 && rm -rf /var/lib/apt/lists/*

RUN conda install -q -y \
        cmake>=3.12 \
        gdal \
        h5py \
        numpy \
        pytest \
        shapely \
        sphinx \
        wget \
        backoff \
 && conda install -q -y -c conda-forge \
        ruamel.yaml \
        yamale \
 && conda clean --all --yes

ARG SRCDIR
ARG BLDDIR
ARG PREFIX

WORKDIR $BLDDIR

RUN echo ". /usr/local/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
 && echo "conda activate base" >> ~/.bashrc
