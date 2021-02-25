ARG runtime_img
FROM $runtime_img

# NB: cuda-cudart-dev package name becomes cuda-cudart-devel for CUDA 11
RUN yum install -y \
        cuda-cudart-dev-$CUDA_PKG_VERSION \
        cuda-cufft-dev-$CUDA_PKG_VERSION \
        cuda-nvcc-$CUDA_PKG_VERSION \
        fftw-devel \
        fftw-libs \
        gcc-c++ \
        gdal-devel \
        git \
        hdf5-devel \
        ninja-build \
        python3-devel \
        python3-pybind11 \
        python3-pytest \
        rpm-build \
 && rm -rf /var/cache/yum/*

# Centos 8 is stuck on CMake 3.11, even in the EPEL!
# https://bugzilla.redhat.com/show_bug.cgi?id=1756974
RUN curl -OSsL https://github.com/Kitware/CMake/releases/download/v3.18.0/cmake-3.18.0-Linux-x86_64.sh \
 && bash cmake-*.sh --prefix=/usr --skip-license \
 && rm cmake-*.sh

# Centos 8 EPEL shapely is too old
RUN pip3 install shapely
