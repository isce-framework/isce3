# Use specific version of nvidia/cuda:11.0-base-centos7 image
FROM nvidia/cuda@sha256:260c6346ca819adcfb71993ad44c0d0623ab93ce6df67567eec9d2278da07802

# Trying to install a package that doesn't exist should be an error.
RUN yum update -y \
 && yum clean all \
 && echo 'skip_missing_names_on_install=False' >> /etc/yum.conf \
 && rm -rf /var/cache/yum

# install latest miniconda
ARG conda_prefix
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.9.2-Linux-x86_64.sh \
        -o miniconda.sh \
 && bash miniconda.sh -b -p $conda_prefix \
 && rm miniconda.sh
ENV PATH="$conda_prefix/bin:$PATH"

COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -tipsy \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

# set up conda environment
RUN echo ". $conda_prefix/etc/profile.d/conda.sh" >> /etc/bashrc \
 && echo "conda activate base"                    >> /etc/bashrc
ENV GDAL_DATA=$conda_prefix/share/gdal
ENV GDAL_DRIVER_PATH=$conda_prefix/lib/gdalplugins
ENV PROJ_LIB=$conda_prefix/share/proj
ENV MPLCONFIGDIR=/tmp

ENV CUDA_PKG_VERSION 11-0

RUN yum install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
        libcufft-$CUDA_PKG_VERSION \
 && yum clean all \
 && rm -rf /var/cache/yum
