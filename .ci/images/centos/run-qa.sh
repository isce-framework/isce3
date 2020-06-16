#!/bin/bash

if [ "$#" -ne 4 ]; then
  echo "Enter args: $0 <tag> <pge> <data_dir> <input_data>"
  echo "e.g.: $0 my-test-tag verify_slc.py `pwd` rslc.h5"
  exit 1
fi

TAG=$1
PGE=$2
DATADIR=$3
DATA=$4    # must be inside DATADIR directory
IMAGE=nisar/isce-ops-qa

CONTAINERTAG=qa-${TAG}
CONTAINER_DATADIR=/home/conda/data 

echo "IMAGE is $IMAGE"
echo "IMAGE TAG is $TAG"
echo "CONTAINER TAG is $CONTAINERTAG"

###Run the container
docker run \
  --rm \
  -u $UID:$(id -g) \
  -v ${DATADIR}:${CONTAINER_DATADIR} \
  -w ${CONTAINER_DATADIR} \
  --name ${CONTAINERTAG} ${IMAGE}:${TAG} \
  /bin/bash -c "source /opt/docker/bin/entrypoint_source &&
                python3 /opt/QualityAssurance/${PGE} --fpdf graphs.pdf --fhdf stats.h5 --flog log.txt --validate --quality ${DATA}"

###Delete the container
docker rm ${CONTAINERTAG}
