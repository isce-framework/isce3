#!/bin/bash
set -ex

#Get the tag from the end of the GIT_BRANCH
ISCEBRANCH="${GIT_BRANCH##*/}"

#Get repo path by removing http://*/ and .git from GIT_URL
REPO="${GIT_URL#*://*/}"
REPO="${REPO%.git}"
#REPO="${REPO//\//_}"

echo "ISCE BRANCH: $ISCEBRANCH"
echo "REPO: $REPO"
echo "WORKSPACE: $WORKSPACE"
echo "GIT_OAUTH_TOKEN: $GIT_OAUTH_TOKEN"

#Get tag
TAG="$(date -u +%Y%m%d)-WFPROFILE"
echo "TAG: $TAG"

# prune docker
docker system prune -f

# turn off valgrind
memcheck=0

# build base and dev images
cd .ci/images/centos
./build-base.sh ${TAG}
curl -L -H "Accept: application/octet-stream" --output gcc7.tar.gz "https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/api/v3/repos/NISAR-ADT/gcc7/releases/assets/40"
./build-dev.sh ${TAG}

# use multistage docker build to build isce from source
# in dev container, run tests and memory checks, create
# rpm, and install in the minimal base image
./build-isce-src.sh ${TAG} ${WORKSPACE}
./build-isce-ops.sh ${TAG} ${WORKSPACE}

git clone --single-branch --branch ${ISCEBRANCH} \
  https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/isce-3/isce.git 

git clone --single-branch \
  https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/NISAR-ADT/WorkflowProfile.git

./build-profile.sh  ${TAG} ${WORKSAPCE} WorkflowProfile

# download test data in artifactory 
cd /tmp
curl -u gmanipon:${ART_CREDENTIALS} -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/r0/ISCE/winnip_data.tar.gz"
tar xvzf winnip_data.tar.gz
cd -
./run-profile.sh ${TAG} /tmp/winnip_data
docker image rm nisar/profile:${TAG}
