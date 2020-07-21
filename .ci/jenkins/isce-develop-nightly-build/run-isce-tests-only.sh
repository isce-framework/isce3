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

# turn on valgrind
memcheck=1

# run tests and memory checks
./run-isce-tests.sh ${TAG} $memcheck
