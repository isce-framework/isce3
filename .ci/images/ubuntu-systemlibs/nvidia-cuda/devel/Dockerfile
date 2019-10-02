FROM nvidia/cuda:10.1-base-ubuntu19.04
LABEL maintainer "NVIDIA CORPORATION <cudatools@nvidia.com>"

RUN apt-get update && apt-get install -y --no-install-recommends \
        cuda-libraries-dev-$CUDAVER \
        cuda-minimal-build-$CUDAVER \
        cuda-command-line-tools-$CUDAVER

ENV LIBRARY_PATH /usr/local/cuda/lib64/stubs
