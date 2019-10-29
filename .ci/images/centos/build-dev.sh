#!/bin/bash

if [ "$#" -ne 1 ]; then
  echo "Enter tag as arg: $0 <tag>"
  echo "e.g.: $0 20180906"
  echo "      $0 my-test-tag"
  exit 1
fi

set -ex

IMAGE=nisar/dev
echo "IMAGE is $IMAGE"

docker build --rm --force-rm -t ${IMAGE}:latest -f Dockerfile.dev .
