#!/bin/bash
# must build all previous dependencies first
# cd /scratch/yzheng/tools/isce/src/isce/.ci/images/cento

if [ "$#" -lt 2 ]; then
  echo "Enter tag as arg: $0 <options> <tag> <isce src> \\"
  echo "options: -g <path to gcc tar.gz to copy under ISCE/.ci/images/centos> \\"
  echo "         -p <path to workflow profile code> \\"
  echo "         -q <path to QualityAssurance code> "
  echo
  echo "e.g.: $0 -g /scratch/yzheng/tools/gcc7/gcc7.tar.gz \\"
  echo "         -p /scratch/yzheng/src/WorkflowProfile \\"
  echo "         -q /scratch/yzheng/tools/QualityAssurance \\"
  echo "         workflow /scratch/yzheng/tools/isce/src/isce "
  exit 1
fi

while getopts 'g:p:q:' opt; do
  case $opt in
    g) GCCTAR=$OPTARG ;;
    p) WFPROFILE=$OPTARG ;;
    q) QA=$OPTARG ;;
  esac
done
shift $(($OPTIND - 1))

TAG=$1
ISCE=$2

echo TAG:       $TAG
echo ISCE:      $ISCE
echo GCCTAR:    $GCCTAR
echo QA:        $QA
echo WFPROFILE: $WFPROFILE

# need to have isce3 setup already and sourced the conda environment
# to ISCE3 directory to do docker builds
pushd ${ISCE}/.ci/images/centos
if [ -n "${GCCTAR+set}" ]; then
  cp ${GCCTAR} .  # needed by build-dev.sh
fi  
./build-base.sh ${TAG}
./build-dev.sh ${TAG}
./build-isce-src.sh ${TAG} ${ISCE} 
./build-isce-ops.sh ${TAG} ${ISCE}

# build ISCE3 image including QA software
if [ -n "${QA+set}" ]; then 
  ./build-isce-ops-qa.sh ${TAG} ${ISCE} ${QA}
fi

# build workflow profile software in image
if [ -n "${WFPROFILE+set}" ]; then
  ./build-profile.sh ${TAG} ${ISCE} ${WFPROFILE}
fi


