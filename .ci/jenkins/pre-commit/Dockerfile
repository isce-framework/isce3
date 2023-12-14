FROM mambaorg/micromamba
ENV PRE_COMMIT_HOME=/tmp/pre-commit
ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN micromamba install -y -c conda-forge pre-commit
USER root
RUN apt update && apt install -y git
USER $MAMBA_USER
