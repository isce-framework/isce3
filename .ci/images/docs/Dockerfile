FROM ubuntu
RUN apt-get update && apt-get install -y \
    doxygen \
    python3-dev \
    python3-sphinx && \
    rm -rf /var/lib/apt/lists/*

# set up permissions
ARG UNAME=user
ARG UID=1000
ARG GID=1000

ARG SRCDIR
ARG DOCDIR

RUN groupadd -g $GID -o $UNAME \
 && useradd -m -u $UID -g $GID -o -s /bin/bash $UNAME \
 && mkdir -p $DOCDIR && chown $UID:$GID $DOCDIR

VOLUME $DOCDIR
USER $UNAME

WORKDIR $SRCDIR
