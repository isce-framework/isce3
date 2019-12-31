#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Enter args: $0 <tag> <memcheckflag>"
  echo "e.g.: $0 20180906 0"
  echo "      $0 my-test-tag 1"
  exit 1
fi

set -ex

TAG=$1
MEMCHECK=$2
IMAGE=nisar/isce-src
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

CONTAINERTAG=isce-test-latest-${TAG}

if [ "$MEMCHECK" = "1" ]; then
    TESTNAME="Test"
else
    TESTNAME="MemCheck"
fi

###Run the container
if [ "$MEMCHECK" = "1" ]; then
   cat run-memcheck.sh | nvidia-docker run --name ${CONTAINERTAG} ${IMAGE}:${TAG} /bin/bash -c 'cat'
else
   cat run-test.sh | nvidia-docker run --name ${CONTAINERTAG} ${IMAGE}:${TAG} /bin/bash -c 'cat'
fi

###Copy file out of the container
docker cp ${CONTAINERTAG}:/home/conda/build/Test.xml .
docker cp ${CONTAINERTAG}:/home/conda/build/doc/html doc
docker cp ${CONTAINERTAG}:/home/conda/build/cppcheck.xml .
if [ "$MEMCHECK" = "1" ]; then
   docker cp  ${CONTAINERTAG}:/home/conda/build/DynamicAnalysis.xml .
   docker cp  ${CONTAINERTAG}:/home/conda/build/valgrind/. .

   #Remove timeouts from valgrind run
   set +e
   for xx in `ls memcheck.*.xml`;
   do
       `xmllint --noout $xx 2>/dev/null`
       xmlflag=$?
       if [ $xmlflag -ne 0 ];
       then
           rm $xx
       fi
   done
   set -e
fi

###Delete the container
docker rm ${CONTAINERTAG}

