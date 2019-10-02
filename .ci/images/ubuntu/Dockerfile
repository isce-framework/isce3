FROM ubuntu:18.04

MAINTAINER Piyush Agram <piyush.agram@jpl.nasa.gov>

# Set an encoding to make things work smoothly.
ENV LANG en_US.UTF-8

#Set non-interactive environment
ENV DEBIAN_FRONTEND noninteractive

# Install devtoolset 6.
RUN apt update  && \
    apt install -y --no-install-recommends \
                gnupg2 \
                curl \
                ca-certificates \
                bzip2 \
                gcc \
                g++ && \
    rm -rf /var/lib/apt/lists/*

# give sudo permission for conda user to run apt (user creation is postponed
# to the entrypoint, so we can create a user with the same id as the host)
RUN echo 'conda ALL=NOPASSWD: /usr/bin/apt' >> /etc/sudoers

# Install the latest Miniconda with Python 3 and update everything.
COPY requirements.txt /tmp/
RUN curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -o miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh && \
    touch /opt/conda/conda-meta/pinned && \
    export PATH=/opt/conda/bin:$PATH && \
    conda config --set show_channel_urls True && \
    conda config --add channels conda-forge && \
    conda update --all --yes && \
    conda install --yes --file /tmp/requirements.txt && \
    conda clean -tipy && \
    rm /tmp/requirements.txt


# Install docker tools.
COPY gosutini.sh /tmp/
RUN bash /tmp/gosutini.sh && \
    rm /tmp/gosutini.sh

# Add a file for users to source to activate the `conda`
# environment `root` and the devtoolset compiler. Also
# add a file that wraps that for use with the `ENTRYPOINT`.
COPY entrypoint_source /opt/docker/bin/entrypoint_source
COPY entrypoint /opt/docker/bin/entrypoint

# Ensure that all containers start with tini and the user selected process.
# Activate the `conda` environment `root` and the devtoolset compiler.
# Provide a default command (`bash`), which will start if the user doesn't specify one.
ENTRYPOINT [ "/opt/conda/bin/tini", "--", "/opt/docker/bin/entrypoint" ]
CMD [ "/bin/bash" ]
