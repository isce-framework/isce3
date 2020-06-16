#!/bin/bash

if [ "$#" -ne 3 ]; then
  echo "Enter args: $0 <tag> <ISCE install dir> <cloned QualityAssurance repo>"
  echo "e.g.: $0 20180906 /home/gmanipon/dev/isce /home/gmanipon/dev/WorkFlowProfile"
  echo "      $0 my-test-tag /scratch/yzheng/tools/isce/src/isce /scratch/yzheng/tools/QualityAssurance"
  exit 1
fi

set -ex

TAG=$1
ISCEDIR=$2
QA=$3
IMAGE=nisar/isce-ops-qa
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

# check Docker.isce-ops-qa exisits
if [ ! -e "${ISCEDIR}/.ci/images/centos/Dockerfile.isce-ops-qa" ]; then
  echo "The Docker file doesn't exist at ${ISCEDIR}/.ci/images/centos/Dockerfile.isce-ops-qa"
  echo "Ensure that it exists by installing it under ${ISCEDIR}/.ci/images/centos/."
  exit 1
fi

# check QualityAssurance directory exisits
if [ ! -d "${QA}" ]; then
  echo "The QualityAssurance repo doesn't exist at ${QA}."
  echo "Ensure that it exists by cloning it under ${QA}."
  exit 1
fi

# replace TAG with correct isce-ops-qa tag
sed -i "s/__TAG__/${TAG}/" ${ISCEDIR}/.ci/images/centos/Dockerfile.isce-ops-qa

docker build --rm --force-rm -t ${IMAGE}:${TAG} \
  -f ${ISCEDIR}/.ci/images/centos/Dockerfile.isce-ops-qa ${QA}
