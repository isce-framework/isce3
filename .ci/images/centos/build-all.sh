#!/bin/bash
# must build all previous dependencies first
# cd /scratch/yzheng/tools/isce/src/isce/.ci/images/cento

if [ "$#" -ne 5 ]; then
  echo "Enter tag as arg: $0 <tag> <isce src> <workflowprofile src> <gcc tar path> <data dir>"
  echo "e.g.: $0 workflow /scratch/yzheng/tools/isce/src/isce \\"
  echo "                  /scratch/yzheng/src/WorkflowProfile \\"
  echo "                  /scratch/yzheng/tools/gcc7/gcc7.tar.gz \\"
  echo "                  /scratch/yzheng/data/winnip_r2.docker3"
  exit 1
fi

TAG=$1
ISCE=$2
WFPROFILE=$3
GCCTAR=$4
DATADIR=$5

# need to have isce3 setup already and sourced the conda environment
# to ISCE3 directory to do docker builds
pushd ${ISCE}/.ci/images/centos
cp ${GCCTAR} .  # needed by build-dev.sh
./build-base.sh ${TAG}
./build-dev.sh ${TAG}
./build-isce-src.sh ${TAG} ${ISCE} 
./build-isce-ops.sh ${TAG} ${ISCE}
./build-profile.sh  ${TAG} ${ISCE} ${WFPROFILE}
rm ${GCCTAR}

## tar up docker file
#docker save -o profile_r0.tar nisar/profile:${TAG}
## load on run docker on another machine
#docker load --input profile_r0.tar

# run test
popd
#cd /scratch/yzheng/test/docker # example test directory
${ISCE}/.ci/images/centos/run-profile.sh ${TAG} ${DATADIR}

# cleanup
docker image rm nisar/profile:${TAG}

