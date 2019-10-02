#!/bin/bash
set -ex

#Get the tag from the end of the GIT_BRANCH
BRANCH="${GIT_BRANCH##*/}"

#Get repo path by removing http://*/ and .git from GIT_URL
REPO="${GIT_URL#*://*/}"
REPO="${REPO%.git}"

TAG="$(date -u +%Y%m%d)-PR-${BUILD_ID}"

echo "BRANCH:    $BRANCH"
echo "REPO:      $REPO"
echo "WORKSPACE: $WORKSPACE"
echo "TAG:       $TAG"

###Replace TAG with correct isce-src tag
sed -i "s/__TAG__/${TAG}/" ${WORKSPACE}/isceCI/images/ubuntu-systemlibs/Dockerfile.isce-release

docker build . -t nisar/cu1904-release:$TAG \
    -f $WORKSPACE/isceCI/images/ubuntu-systemlibs/Dockerfile.isce-release


#Clean up new images
#For now, till we have artifactory setup
docker rmi nisar/cu1904-release:$TAG
docker images -q --filter "dangling=true" --filter "label=cmakelabel=${TAG}" | xargs -r docker rmi
