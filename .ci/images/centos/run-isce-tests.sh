#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Enter args: $0 <tag> <memcheckflag>"
  echo "e.g.: $0 20180906 0"
  echo "      $0 my-test-tag 1"
  exit 1
fi
set -exu

TAG=$1
MEMCHECK=$2
IMAGE=nisar/isce-src
echo "IMAGE is $IMAGE"
echo "TAG is $TAG"

CONTAINERTAG=isce-test-latest-${TAG}

if [ "$MEMCHECK" = "0" ]; then
    TESTNAME="Test"
else
    TESTNAME="MemCheck"
fi

# Run the container
if [ "$MEMCHECK" = "0" ]; then
    docker run --name ${CONTAINERTAG} ${IMAGE}:${TAG} /bin/bash -ex -c \
        'source /opt/docker/bin/entrypoint_source \
          && cd build \
          && ctest -j `nproc` --no-compress-output --output-on-failure -T Test || true \
          && cp Testing/$(head -1 Testing/TAG)/Test.xml .'
else
    docker run --name ${CONTAINERTAG} ${IMAGE}:${TAG} /bin/bash -ex -c \
        'source /opt/docker/bin/entrypoint_source \
          && cd build \
          && ctest --no-compress-output --output-on-failure -T Test || true \
          && cp Testing/$(head -1 Testing/TAG)/Test.xml . \
          && ctest --no-compress-output --output-on-failure --timeout 10000 -T MemCheck \
                -E "(iscecuda.core.stream|backproject|workflows)" \
                || true \
          && cp Testing/$(head -1 Testing/TAG)/DynamicAnalysis.xml .'
fi

# boilerplate documentation paths copied from Jenkins PR run script
BLDDIR=/home/conda/build
SRCDIR=/home/conda/isce
SPHX_SRC=$SRCDIR/doc/sphinx
SPHX_CONF=$BLDDIR/doc/sphinx
SPHX_DIR=$BLDDIR/docs-output/sphinx
SPHX_CACHE=$SPHX_DIR/_doctrees
SPHX_HTML=$SPHX_DIR/html

DOCSTAG=isce-centos-docs-latest-$TAG
docker run --name $DOCSTAG $IMAGE:$TAG bash -exc \
    "PYTHONPATH=$BLDDIR/packages/isce3/extensions \
     sphinx-build -q -b html -c $SPHX_CONF -d $SPHX_CACHE $SPHX_SRC $SPHX_HTML \
     && doxygen $BLDDIR/doc/doxygen/Doxyfile"
docker cp $DOCSTAG:$BLDDIR/docs-output .
docker rm $DOCSTAG

###Copy file out of the container
docker cp ${CONTAINERTAG}:/home/conda/build/Test.xml .
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
