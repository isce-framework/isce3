#!/bin/bash
set -ex

if [ "$#" -eq 1 ]; then
  WORKSPACE=$1
fi

IMAGE=nisar/isce-ops-qa
# Get tag
TAG="test-gslc-$(date -u +%Y%m%d)-${BUILD_NUMBER}-${GIT_COMMIT:0:10}"
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
  "https://${GIT_OAUTH_TOKEN}@github-fn.jpl.nasa.gov/NISAR-ADT/QualityAssurance.git"
./build-base.sh ${TAG}
./build-dev.sh ${TAG}
./build-isce-src.sh ${TAG} ${WORKSPACE}
./build-isce-ops.sh ${TAG} ${WORKSPACE}
./build-isce-ops-qa.sh ${TAG} ${WORKSPACE} QualityAssurance

DATADIR=${WORKSPACE}/test_gcov
CONTAINERTAG=container-${TAG}
CONTAINER_DATADIR=/tmp/data

# download test data from artifactory 
if [ -d ${DATADIR} ]; then
  rm -rf ${DATADIR}
fi
mkdir -p ${DATADIR}
cd ${DATADIR}
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/GSLC_GCOV_test_SanAnd/run_config_gslc.yaml"
mkdir input
cd input
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/GSLC_GCOV_test_SanAnd/input/SanAnd_05024_18038_006_180730_L090_CX_129_05.h5"
mkdir -p nisar-dem/EPSG32610
cd nisar-dem/EPSG32610
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/GSLC_GCOV_test_SanAnd/input/nisar-dem/EPSG32610/EPSG32610.vrt"
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/GSLC_GCOV_test_SanAnd/input/nisar-dem/EPSG32610/N4000E0400.tif"
curl -O "https://cae-artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/test/GSLC_GCOV_test_SanAnd/input/nisar-dem/EPSG32610/N4000E0600.tif"

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
                mkdir output_gslc scratch_gslc qa_gslc &&
                time python3 /opt/isce/packages/nisar/workflows/rslc2gslc.py run_config_gslc.yaml &&
                time python3 /opt/QualityAssurance/verify_gslc.py  --fpdf qa_gslc/graphs.pdf --fhdf qa_gslc/stats.h5 --flog qa_gslc/qa.log --validate --quality output_gslc/gslc.h5
               "

# check if output files exist
for file in output_gslc/gslc.h5 qa_gslc/graphs.pdf qa_gslc/stats.h5 qa_gslc/qa.log; do
  if [[ (! -f ${DATADIR}/${file}) ]]; then
    echo "Error: expected output GSLC file not found: ${file}"
    exit 1
  fi
done
cat ${DATADIR}/qa_gslc/qa.log
# TODO: check for correct file size, run time, CF convention, 
