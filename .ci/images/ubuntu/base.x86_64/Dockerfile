FROM ubuntu:19.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

# TODO use 19.04 repo once available
ENV NVIDIA_PKG_URL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64

RUN apt-get update && apt-get install -y --no-install-recommends \
        gnupg2 curl ca-certificates && \
    curl -fsSL $NVIDIA_PKG_URL/7fa2af80.pub | apt-key add - && \
    echo "deb $NVIDIA_PKG_URL /" > /etc/apt/sources.list.d/cuda.list && \
    apt-get purge --autoremove -y curl && \
    rm -rf /var/lib/apt/lists/*

ENV CUDAVER 10-2

# For libraries in the cuda-compat-* package:
# https://docs.nvidia.com/cuda/eula/index.html#attachment-a
RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-cudart-$CUDAVER \
        cuda-compat-$CUDAVER && \
    ln -s cuda-10.2 /usr/local/cuda && \
    rm -rf /var/lib/apt/lists/*

# Required for nvidia-docker v1
RUN echo "/usr/local/nvidia/lib"   >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

# nvidia-container-runtime
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=10.2 brand=tesla,driver>=384,driver<385 brand=tesla,driver>=396,driver<397 brand=tesla,driver>=410,driver<411 brand=tesla,driver>=418,driver<419"
