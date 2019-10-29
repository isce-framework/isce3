#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Enter tag as arg: $0 <tag>"
  echo "e.g.: $0 20180906"
  echo "      $0 my-test-tag"
  exit 1
fi

IMAGE=nisar/base
echo "IMAGE is $IMAGE"

# fail on any non-zero exit codes
set -ex

# pull latest version of base images
docker pull nvidia/cuda:9.2-runtime-centos7
docker tag nvidia/cuda:9.2-runtime-centos7 nvidia/cuda:latest

# build base cuda image
docker build --rm --force-rm -t ${IMAGE}:latest -f Dockerfile.base .
