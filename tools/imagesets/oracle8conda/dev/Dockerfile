ARG runtime_img
FROM $runtime_img

# Install dependencies for ISCE development
RUN set -ex \
 && yum update -y \
 && yum install -y \
        cuda-cudart-devel-$CUDA_PKG_VERSION \
        cuda-nvcc-$CUDA_PKG_VERSION \
        rpm-build \
 && yum clean all \
 && rm -rf /var/cache/yum \
 && rm -rf /var/cache/yum

# The libcufft-devel-11-8 RPM is missing SHA256 digests, so the usual `yum install`
# command fails on some machines with 'package <...> does not verify: no digest'. See
# https://forums.developer.nvidia.com/t/unable-to-install-cuda-rpms-and-libraries-in-rhel8-fips-mode-no-digest/230008
# for details. Yum doesn't appear to support an equivalent to
# `rpm --install --nodigest`, so we need this roundabout approach of using yum to
# download the RPM(s) and then using rpm to install the package(s). This seems to be
# resolved in CUDA 12, so we could revert this when we upgrade from CUDA 11.
RUN yum install -y --downloadonly --destdir /tmp \
        libcufft-devel-$CUDA_PKG_VERSION \
 && rpm --install --nodigest /tmp/*.rpm \
 && rm /tmp/*.rpm

COPY spec-file.txt /tmp/spec-file.txt
RUN conda install --yes --file /tmp/spec-file.txt \
 && conda clean -ay \
 && rm -rf /opt/conda/pkgs \
 && rm /tmp/spec-file.txt

ENV CUDAHOSTCXX=x86_64-conda-linux-gnu-c++
ENV CUDACXX=/usr/local/cuda-${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}/bin/nvcc
