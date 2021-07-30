#!/bin/bash
set -ex

if [ "$#" -eq 1 ]; then
  WORKSPACE=$1
fi

IMAGE=nisar/isce-ops-qa
# Get tag
TAG="test-rslc-$(date -u +%Y%m%d)-${BUILD_NUMBER}-${GIT_COMMIT:0:10}"
# Get the tag from the end of the GIT_BRANCH
ISCEBRANCH="${GIT_BRANCH##*/}"

echo "Building ISCE3 Docker image ..."
echo "IMAGE: $IMAGE"
echo "BRANCH: $ISCEBRANCH"
echo "TAG: $TAG"
echo "WORKSPACE: $WORKSPACE"

# cleanup unused Docker resources
docker system prune -f

# build Docker image containing ISCE3 and QA
cd ${WORKSPACE}/.ci/images/centos
curl -L -H "Accept: application/octet-stream" --output gcc7.tar.gz \
  "https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/api/v3/repos/NISAR-ADT/gcc7/releases/assets/40"
git clone --single-branch \
  https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance.git
./build-base.sh ${TAG}
./build-dev.sh ${TAG}
./build-isce-src.sh ${TAG} ${WORKSPACE}
./build-isce-ops.sh ${TAG} ${WORKSPACE}
./build-isce-ops-qa.sh ${TAG} ${WORKSPACE} QualityAssurance

DATADIR=${WORKSPACE}/test_rslc
CONTAINERTAG=container-${TAG}
CONTAINER_DATADIR=/tmp/data

# download test data from artifactory 
if [ -d ${DATADIR} ]; then
  rm -rf ${DATADIR}
fi
mkdir -p ${DATADIR}
cd ${DATADIR}
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/RSLC_test_REE1/run_config_rslc.yaml"
mkdir input
cd input
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/RSLC_test_REE1/input/REE_L0B_out17.h5"

# run actual test
cd ${WORKSPACE}/tests/workflows
docker run \
  --rm \
  -u $UID:$(id -g) \
  --mount type=bind,source=${DATADIR},target=${CONTAINER_DATADIR} \
  -w ${CONTAINER_DATADIR} \
  --name ${CONTAINERTAG} ${IMAGE}:${TAG} \
  /bin/bash -c "source /opt/docker/bin/entrypoint_source &&
                echo ${CONTAINER_DATADIR}: && 
                ls -al ${CONTAINER_DATADIR} && 
                cd ${CONTAINER_DATADIR} && 
                mkdir output_rslc scratch_rslc qa_rslc &&
                time python3 /opt/isce/packages/nisar/workflows/focus.py run_config_rslc.yaml &&
                time python3 /opt/QualityAssurance/verify_rslc.py  --fpdf qa_rslc/graphs.pdf --fhdf qa_rslc/stats.h5 --flog qa_rslc/qa.log --validate --quality output_rslc/rslc.h5
               "

# check if output files exist
for file in output_rslc/rslc.h5 output_rslc/rslc_config.yaml qa_rslc/graphs.pdf qa_rslc/stats.h5 qa_rslc/qa.log; do
  if [[ (! -f ${DATADIR}/${file}) ]]; then
    echo "Error: expected output RSLC file not found: ${file}"
    exit 1
  fi
done
cat ${DATADIR}/qa_rslc/qa.log
# TODO: check for correct file size, run times, CF convention.
