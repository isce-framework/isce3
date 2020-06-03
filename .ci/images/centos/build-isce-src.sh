#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Enter args: $0 <tag> <cloned ISCE repo>"
  echo "e.g.: $0 20180906 /home/gmanipon/dev/isce"
  echo "      $0 my-test-tag /home/gmanipon/dev/isce"
  exit 1
fi

set -ex

TAG=$1
WORKSPACE=$2
IMAGE=nisar/isce-src
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

# check .ci directory exists
if [ ! -d "${WORKSPACE}/.ci" ]; then
  echo "Error: the .ci directory doesn't exist at ${WORKSPACE}/.ci"
  exit 1
fi

docker build --rm --force-rm --network=host -t ${IMAGE}:${TAG}  \
  -f ${WORKSPACE}/.ci/images/centos/Dockerfile.isce-src $WORKSPACE
