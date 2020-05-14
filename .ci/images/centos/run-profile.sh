#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Enter args: $0 <tag> <data_dir>"
  echo "e.g.: $0 workflow /scratch/yzheng/data/winnip"
  echo "      $0 my-test-tag /scrach/lyu/data/sanand"
  exit 1
fi

TAG=$1
DATA=$2
IMAGE=nisar/profile

echo "IMAGE is $IMAGE"
echo "TAG is $TAG"


CONTAINERTAG=profile-${TAG}
CONTAINER_DATADIR=/tmp/data # don't put in home since input data will be owned by centos!!!
CONTAINER_TESTDIR=/home/conda/test

###Run the container
docker run \
  --mount type=bind,source=${DATA},target=${CONTAINER_DATADIR} \
  --name ${CONTAINERTAG} ${IMAGE}:${TAG} \
  /bin/bash -c "source /opt/docker/bin/entrypoint_source &&
                echo ${CONTAINER_DATADIR}: && 
                ls -al ${CONTAINER_DATADIR} && 
                cd ${CONTAINER_TESTDIR} && 
                python3 /opt/WorkflowProfile/WorkflowProfile/workflowprofile.py $CONTAINER_DATADIR/profile_runs.yaml &&
                mv profile_interferogram_*.yaml results.pickle profile_runs"


###Copy file out of the container
docker cp ${CONTAINERTAG}:/home/conda/test/profile_runs .

###Delete the container
docker rm ${CONTAINERTAG}


