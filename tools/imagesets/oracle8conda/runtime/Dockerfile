# Use pinned oraclelinux:8 image
ARG repository=cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux
ARG tag=8.10.241001
ARG digest=sha256:add0de53779191a7afed7ada9549e8fee20ef8486766eb8cb98c0d7639c865f1
FROM ${repository}:${tag}@${digest}

RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Trying to install a package that doesn't exist should be an error.
RUN yum update -y \
 && yum clean all \
 && echo 'skip_missing_names_on_install=False' >> /etc/yum.conf \
 && rm -rf /var/cache/yum

# install latest miniconda
ARG conda_prefix
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py312_24.7.1-0-Linux-x86_64.sh \
        -o miniconda.sh \
 && bash miniconda.sh -b -p $conda_prefix \
 && rm miniconda.sh
ENV PATH="$conda_prefix/bin:$PATH"

# use the libmamba solver, which is faster and has better diagnostics
RUN conda config --set solver libmamba

COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -ay \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

# set up conda environment
RUN echo ". $conda_prefix/etc/profile.d/conda.sh" >> /etc/bashrc \
 && echo "conda activate base"                    >> /etc/bashrc
ENV GDAL_DATA=$conda_prefix/share/gdal
ENV GDAL_DRIVER_PATH=$conda_prefix/lib/gdalplugins
ENV PROJ_LIB=$conda_prefix/share/proj
ENV MPLCONFIGDIR=/tmp

ENV CUDA_VERSION_MAJOR 11
ENV CUDA_VERSION_MINOR 4
ENV CUDA_PKG_VERSION "${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"

RUN yum install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
        libcufft-$CUDA_PKG_VERSION \
 && yum clean all \
 && rm -rf /var/cache/yum

# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}"
