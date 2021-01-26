FROM nvidia/cuda:10.1-base

ENV PATH="/usr/local/conda/bin:${PATH}"

RUN apt-get update && apt-get install -y curl bzip2 \
 && curl -sSL https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-$(arch).sh \
        -o /tmp/miniconda.sh \
 && bash /tmp/miniconda.sh -bfp /usr/local/conda \
 && rm -rf /tmp/miniconda.sh \
 && conda install -q -y python>=3.6 \
 && conda install -q -y -c conda-forge pyre \
 && conda update conda \
 && apt-get -y remove curl bzip2 \
 && apt-get -y autoremove \
 && rm -rf /var/lib/apt/lists/* \
 && conda clean --all --yes \
 && rm -rf /usr/local/conda/pkgs
