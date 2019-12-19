#!/bin/bash
set -xeuo pipefail

if [ "$(id -u)" -eq 0 ]; then
    echo "Error: do not run as root!"
    exit 1
fi

# cd to project root dir
cd `dirname $0`/..

# execute with required environment variables
env BUILD_TAG=`whoami` \
    IMAGE_SUITE=conda \
    WORKSPACE=`pwd` \
    bash .ci/jenkins/develop-conda.sh
