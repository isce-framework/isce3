#!/bin/bash
set -ex

#Get the tag from the end of the GIT_BRANCH
BRANCH="${GIT_BRANCH##*/}"

#Get repo path by removing http://*/ and .git from GIT_URL
REPO="${GIT_URL#*://*/}"
REPO="${REPO%.git}"
#REPO="${REPO//\//_}"

echo "BRANCH: $BRANCH"
echo "REPO: $REPO"
echo "WORKSPACE: $WORKSPACE"
echo "GIT_OAUTH_TOKEN: $GIT_OAUTH_TOKEN"

#Get tag
TAG="$(date -u +%Y%m%d)-NIGHTLY"
echo "TAG: $TAG"

# prune docker
docker system prune -f

# turn on valgrind
memcheck=1

# build base and dev images
cd .ci/images/centos
./build-base.sh ${TAG}
curl -L -H "Accept: application/octet-stream" --output gcc7.tar.gz "https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/api/v3/repos/NISAR-ADT/gcc7/releases/assets/40"

./build-dev.sh ${TAG}
