set -xeuo pipefail

#Get tag
echo "TAG: $BUILD_TAG"

# look for suitable docker executable
# nvidia-docker preferred if available
DOCKER=`which nvidia-docker > /dev/null && \
        echo nvidia-docker || \
        echo docker`

CONTAINER="isce-build-$BUILD_TAG"

SRCDIR=/isce-src # source directory
BLDDIR=/isce-bld # build directory
PREFIX=/opt/isce # install directory
DOCDIR=/isce-docs

DOCKER_BUILD_ARGS="\
    --network=host \
    --build-arg UID=$(id -u) \
    --build-arg GID=$(id -g) \
    --build-arg SRCDIR=$SRCDIR \
    --build-arg BLDDIR=$BLDDIR \
    --build-arg PREFIX=$PREFIX \
    --build-arg DOCDIR=$DOCDIR \
    "

# base & devel cuda images
$DOCKER build $DOCKER_BUILD_ARGS .ci/images/nvidia-cuda/base  -t nvidia/cuda:10.1-base-ubuntu19.04
$DOCKER build $DOCKER_BUILD_ARGS .ci/images/nvidia-cuda/devel -t nvidia/cuda:10.1-devel-ubuntu19.04

# build + testing images
$DOCKER build $DOCKER_BUILD_ARGS .ci/images/ubuntu-builder -t isce-ci/builder
$DOCKER build $DOCKER_BUILD_ARGS .ci/images/ubuntu-tester  -t isce-ci/tester

# build and install isce
$DOCKER rm $CONTAINER || true
$DOCKER run --name $CONTAINER \
    -v `pwd`:$SRCDIR:ro \
    isce-ci/builder bash -c \
    "cmake $SRCDIR -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                   -DCMAKE_INSTALL_PREFIX=$PREFIX \
     && make -j`nproc` VERBOSE=y \
     && make install"

# run cppcheck
CPPCHECK_ARGS="--std=c++14 --enable=all --inconclusive --force --inline-suppr \
               --xml --xml-version=2"
$DOCKER run --rm \
    --volumes-from $CONTAINER \
    isce-ci/tester bash -c \
    "cppcheck $CPPCHECK_ARGS $SRCDIR/cxx 2> $BLDDIR/cppcheck.xml"

# run tests
$DOCKER run --rm \
    --volumes-from $CONTAINER \
    isce-ci/tester bash -c \
    "ctest -j`nproc` -T Test --verbose || true && \
     cp Testing/*/Test.xml ."


# get xml checks/tests
# TODO choose better output dir here
docker cp $CONTAINER:$BLDDIR/Test.xml     .ci/images/centos/
docker cp $CONTAINER:$BLDDIR/cppcheck.xml .ci/images/centos/

#
# Generate documentation using docker image
#

# bail out early if no login credentials provided
if [ -z "${GIT_OAUTH_TOKEN:-}" ]; then exit 1; fi

# documentation builder images
$DOCKER build $DOCKER_BUILD_ARGS .ci/images/docs -t isce-docs

SPHX_SRC=$SRCDIR/doc/sphinx
SPHX_CONF=$BLDDIR/doc/sphinx
SPHX_DIR=$DOCDIR/sphinx
SPHX_CACHE=$SPHX_DIR/_doctrees
SPHX_HTML=$SPHX_DIR/html

$DOCKER run --rm \
    --volumes-from $CONTAINER \
    isce-docs bash -c \
    "PYTHONPATH=$BLDDIR/packages/isce3/extensions \
       sphinx-build -q -b html -c $SPHX_CONF -d $SPHX_CACHE $SPHX_SRC $SPHX_HTML"

$DOCKER run --rm \
    --volumes-from $CONTAINER \
    isce-docs bash -c \
    "doxygen $BLDDIR/doc/doxygen/Doxyfile"

#
# Push documentation to gh-pages
#
docker cp $CONTAINER:$BLDDIR/doc gh-pages
cd gh-pages
mv html/* .
touch .nojekyll

git clone --single-branch --branch gh-pages --no-checkout \
    https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/isce-3/isce tmp
mv tmp/.git .
git add .
git status

git config --local user.name  "gmanipon"
git config --local user.email "gmanipon@jpl.nasa.gov"
git commit -am "auto update of docs ($BUILD_URL)" && git push || echo "no changes committed"
