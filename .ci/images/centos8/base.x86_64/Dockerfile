# Based on https://gitlab.com/nvidia/container-images/cuda/blob/centos7/10.1/base/Dockerfile

FROM centos:8

RUN NVIDIA_GPGKEY_SUM=d1be581509378368edeec8c1eb2958702feedf3bc3d17011adbf24efacce4ab5 \
 && curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/7fa2af80.pub \
        | sed '/^Version/d' > /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA \
 && echo "$NVIDIA_GPGKEY_SUM  /etc/pki/rpm-gpg/RPM-GPG-KEY-NVIDIA" | sha256sum -c --strict -

COPY cuda.repo /etc/yum.repos.d/cuda.repo

ENV CUDA_VERSION 10.2.89
ENV CUDA_PKG_VERSION 10-2-$CUDA_VERSION-1

RUN yum install -y \
        cuda-cudart-$CUDA_PKG_VERSION \
        cuda-compat-10-2 \
 && ln -s cuda-10.2 /usr/local/cuda \
 && rm -rf /var/cache/yum/*

# nvidia-docker 1.0
RUN echo /usr/local/nvidia/lib   >> /etc/ld.so.conf.d/nvidia.conf \
 && echo /usr/local/nvidia/lib64 >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"

ENV PATH /usr/local/conda/bin:$PATH

# Miniconda installation
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-$(arch).sh \
        -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -bfp /usr/local/conda \
 && rm -rf /tmp/miniconda.sh \
 && conda update conda \
 && conda install -qy python \
 && conda clean --all --yes
