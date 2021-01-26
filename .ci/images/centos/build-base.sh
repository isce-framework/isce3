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

# build base cuda image
docker build --rm --force-rm --network=host -t ${IMAGE}:latest -f Dockerfile.base .
