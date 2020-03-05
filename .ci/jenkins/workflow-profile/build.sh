#!/bin/bash
set -ex

#Get the tag from the end of the GIT_BRANCH
ISCEBRANCH="${GIT_BRANCH##*/}"

echo "ISCE BRANCH: $ISCEBRANCH"
echo "WORKSPACE: $WORKSPACE"
echo "GIT_OAUTH_TOKEN: $GIT_OAUTH_TOKEN"

#Get tag
TAG="$(date -u +%Y%m%d)-WFPROFILE"
echo "TAG: $TAG"

# prune docker
docker system prune -f

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

git clone --single-branch \
  https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/NISAR-ADT/WorkflowProfile.git

./build-profile.sh  ${TAG} ${WORKSPACE} WorkflowProfile

# download test data in artifactory 
cd /tmp
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/r0/ISCE/winnip_data.tar.gz"
ls -al winnip_data.tar.gz
tar xvzf winnip_data.tar.gz
cd -
./run-profile.sh ${TAG} /tmp/winnip_data
docker image rm nisar/profile:${TAG}

# check results
if [[ (! -f profile_runs/test_gpu.yaml_crossmul.coh) || (! -f profile_runs/test_gpu.yaml_crossmul.int) ]]; then
  echo "Error: no GPU interferogram or coherence results found"
  exit 1
fi
