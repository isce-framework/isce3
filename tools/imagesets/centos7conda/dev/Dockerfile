ARG runtime_img
FROM $runtime_img

# Install dependencies for ISCE developement
RUN set -ex \
 && yum update -y \
 && yum install -y \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-nvcc-$CUDA_PKG_VERSION \
        doxygen \
        ninja \
        rpm-build \
        sphinx \
 && yum clean all \
 && rm -rf /var/cache/yum \
 && rm -rf /var/cache/yum

# Prefer installing from main, as forge is slow to resolve dependencies.
COPY requirements.txt.main.dev  /tmp/requirements.txt.main
COPY requirements.txt.forge.dev /tmp/requirements.txt.forge
RUN conda install --yes -c main        --file /tmp/requirements.txt.main  \
 && conda install --yes -c conda-forge --file /tmp/requirements.txt.forge \
 && conda clean -tipsy \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/requirements.txt.{main,forge}

ENV CUDAHOSTCXX=x86_64-conda_cos6-linux-gnu-g++
