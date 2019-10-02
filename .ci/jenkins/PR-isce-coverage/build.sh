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

IMAGENAME="nisar/isce-cu1904-coverage"

###Replace TAG with correct isce-src tag
sed -i "s/__TAG__/${TAG}/" ${WORKSPACE}/isceCI/images/ubuntu-systemlibs/Dockerfile.isce-coverage

nvidia-docker build . -t $IMAGENAME:$TAG \
    -f $WORKSPACE/isceCI/images/ubuntu-systemlibs/Dockerfile.isce-coverage

# Get the coverage output XML for Cobertura
nvidia-docker run --rm $IMAGENAME:$TAG cat /coverage.xml > coverage.xml

#Clean up the image created
docker rmi $IMAGENAME:$TAG
docker images -q --filter "dangling=true" --filter "label=cmakelabel=${TAG}" | xargs -r docker rmi
