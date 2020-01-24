#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Enter args: $0 <tag> <ISCE install dir> <cloned WorkFlowProfile repo>"
  echo "e.g.: $0 20180906 /home/gmanipon/dev/isce /home/gmanipon/dev/WorkFlowProfile"
  echo "      $0 my-test-tag /home/gmanipon/dev/isce /home/gmanipon/dev/WorkFlowProfile"
  exit 1
fi

set -ex

TAG=$1
ISCEDIR=$2
WFPROFILE=$3
IMAGE=nisar/profile
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

# check Docker.profile exisits
if [ ! -e "${ISCEDIR}/.ci/images/centos/Dockerfile.profile" ]; then
  echo "The file Dockerfile.profile doesn't exist at ${ISCEDIR}/.ci/images/centos/Dockerfile.profile"
  echo "Ensure that it exists by installing it under ${ISCEDIR}."
  exit 1
fi

# check WorkFlowProfile directory exisits
if [ ! -d "${WFPROFILE}" ]; then
  echo "The WorkFlowProfile repo doesn't exist at ${WFPROFILE}."
  echo "Ensure that it exists by cloning it under ${WFPROFILE}."
  exit 1
fi

#Replace TAG with correct isce-src tag
sed -i "s/__TAG__/${TAG}/" ${ISCEDIR}/.ci/images/centos/Dockerfile.profile

docker build --rm --force-rm -t ${IMAGE}:${TAG} \
  -f ${ISCEDIR}/.ci/images/centos/Dockerfile.profile ${WFPROFILE}
#docker tag ${IMAGE}:${TAG} ${IMAGE}:latest
