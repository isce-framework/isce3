FROM nvidia/cuda:11.0-base-centos7

# Trying to install a package that doesn't exist should be an error.
RUN yum update -y \
 && yum clean all \
 && echo 'skip_missing_names_on_install=False' >> /etc/yum.conf \
 && rm -rf /var/cache/yum

# install latest miniconda
ARG conda_prefix
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
        -o miniconda.sh \
 && bash miniconda.sh -b -p $conda_prefix \
 && rm miniconda.sh
ENV PATH="$conda_prefix/bin:$PATH"

# Prefer installing from main, as forge is slow to resolve dependencies.
COPY requirements.txt.main.runtime  /tmp/requirements.txt.main
COPY requirements.txt.forge.runtime /tmp/requirements.txt.forge
RUN conda install --yes -c main        --file /tmp/requirements.txt.main  \
 && conda install --yes -c conda-forge --file /tmp/requirements.txt.forge \
 && conda clean -tipsy \
 && rm -rf $conda_prefix/pkgs \
 && rm /tmp/requirements.txt.{main,forge}

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
