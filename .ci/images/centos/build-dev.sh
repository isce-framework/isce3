#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Enter tag as arg: $0 <tag>"
  echo "e.g.: $0 20180906"
  echo "      $0 my-test-tag"
  exit 1
fi

set -ex

TAG=$1
IMAGE=nisar/dev
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

docker build --rm --force-rm -t ${IMAGE}:${TAG} -f Dockerfile.dev .
docker tag ${IMAGE}:${TAG} ${IMAGE}:latest
