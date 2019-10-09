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
TAG=$(date -u +%Y%m%d)
echo "TAG: $TAG"

# turn off valgrind
memcheck=0

# build base and dev images
cd .ci/images/centos
./build-base.sh $TAG
curl -L -H "Accept: application/octet-stream" --output gcc7.tar.gz \
    "https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/api/v3/repos/NISAR-ADT/gcc7/releases/assets/40"
./build-dev.sh $TAG

# use multistage docker build to build isce from source
# in dev container, run tests and memory checks, create
# rpm, and install in the minimal base image
./build-isce-src.sh $TAG $WORKSPACE
./run-isce-tests.sh $TAG $memcheck
./build-isce-ops.sh $TAG $WORKSPACE

# update gh-pages with latest docs
git clone --single-branch --branch gh-pages \
  https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/isce-3/isce.git 
  cd isce/
  git config user.name "gmanipon"
  git config user.email "gmanipon@jpl.nasa.gov"
  git rm -rf .
  mv ../doc/* .
  touch .nojekyll
  git add .
  git status
  git commit -am "auto update of docs ($BUILD_URL)"
  git push


#Clean up new images
docker rmi $(docker images -q nisar/isce-ops:$TAG)
docker images -q --filter "dangling=true" --filter "label=rpmlabel=${TAG}" | xargs -r docker rmi
docker rmi $(docker images -q nisar/isce-src:$TAG)
