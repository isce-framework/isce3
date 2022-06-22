# Use pinned oraclelinux:8 image
FROM oraclelinux:8.4@sha256:ef0327c1a51e3471e9c2966b26b6245bd1f4c3f7c86d7edfb47a39adb446ceb5

RUN yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo

# Trying to install a package that doesn't exist should be an error.
RUN yum update -y \
 && yum clean all \
 && echo 'skip_missing_names_on_install=False' >> /etc/yum.conf \
 && rm -rf /var/cache/yum

# install latest miniconda
ARG conda_prefix
RUN curl -sSL https://repo.anaconda.com/miniconda/Miniconda3-py39_4.12.0-Linux-x86_64.sh \
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

ENV CUDA_VERSION_MAJOR 11
ENV CUDA_VERSION_MINOR 7
ENV CUDA_PKG_VERSION "${CUDA_VERSION_MAJOR}-${CUDA_VERSION_MINOR}"

RUN yum install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
        libcufft-$CUDA_PKG_VERSION \
 && yum clean all \
 && rm -rf /var/cache/yum

# https://github.com/NVIDIA/nvidia-container-runtime#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.0"
