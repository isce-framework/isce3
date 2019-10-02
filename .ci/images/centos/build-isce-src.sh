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

# check isceCI was cloned
if [ ! -d "${WORKSPACE}/isceCI" ]; then
  echo "The isceCI repo doesn't exist at ${WORKSPACE}/isceCI."
  echo "Ensure that it exists by cloning it under ${WORKSPACE}."
  exit 1
fi
  
nvidia-docker build --rm --force-rm -t ${IMAGE}:${TAG}  \
  -f ${WORKSPACE}/isceCI/images/centos/Dockerfile.isce-src $WORKSPACE
