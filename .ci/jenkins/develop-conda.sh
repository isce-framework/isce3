set -xeuo pipefail

#Get tag
echo "TAG: $BUILD_TAG"

# Use conda images by default
if [ -z "${IMAGE_SUITE:-}" ]; then
    IMAGE_SUITE=conda
fi

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

# create base, build, and testing images
IMAGE_DIR=.ci/images/$IMAGE_SUITE
IMAGE_ID=isce-ci-$IMAGE_SUITE
$DOCKER build $DOCKER_BUILD_ARGS $IMAGE_DIR/base.$(arch) -t $IMAGE_ID/base
$DOCKER build $DOCKER_BUILD_ARGS $IMAGE_DIR/builder      -t $IMAGE_ID/builder
$DOCKER build $DOCKER_BUILD_ARGS $IMAGE_DIR/tester       -t $IMAGE_ID/tester

# build and install isce
$DOCKER rm $CONTAINER || true
$DOCKER run --name $CONTAINER \
    -v `pwd`:$SRCDIR:ro \
    $IMAGE_ID/builder bash -c \
    "cmake $SRCDIR -DWITH_CUDA=y -DCMAKE_BUILD_TYPE=RelWithDebInfo \
                   -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
                   -DCMAKE_INSTALL_PREFIX=$PREFIX \
     && make -j`nproc` VERBOSE=y \
     && make install"

# run cppcheck
CPPCHECK_ARGS="--std=c++14 --enable=all --inconclusive --force --inline-suppr \
               --xml --xml-version=2"
$DOCKER run --rm \
    --volumes-from $CONTAINER \
    $IMAGE_ID/tester bash -c \
    "cppcheck $CPPCHECK_ARGS $SRCDIR/cxx 2> $BLDDIR/cppcheck.xml"

# run tests
$DOCKER run --rm \
    --volumes-from $CONTAINER \
    $IMAGE_ID/tester bash -ic \
    "ctest -j`nproc` -T Test --verbose || true && \
     cp Testing/*/Test.xml ."


# get xml checks/tests
# TODO choose better output dir here
docker cp $CONTAINER:$BLDDIR/Test.xml     $IMAGE_DIR
docker cp $CONTAINER:$BLDDIR/cppcheck.xml $IMAGE_DIR

# copy to old directory to satisfy jenkins
# TODO remove this path once jenkins config is updated
cp $IMAGE_DIR/*.xml .ci/images/centos/

#
# Generate documentation using docker image
#

# bail out early if no login credentials provided
if [ -z "${GIT_OAUTH_TOKEN:-}" ]; then exit 1; fi

# documentation builder
SPHX_SRC=$SRCDIR/doc/sphinx
SPHX_CONF=$BLDDIR/doc/sphinx
SPHX_DIR=$DOCDIR/sphinx
SPHX_CACHE=$SPHX_DIR/_doctrees
SPHX_HTML=$SPHX_DIR/html

$DOCKER run --rm \
    --volumes-from $CONTAINER \
    $IMAGE_ID/tester bash -c \
    "PYTHONPATH=$BLDDIR/packages/isce3/extensions \
       sphinx-build -q -b html -c $SPHX_CONF -d $SPHX_CACHE $SPHX_SRC $SPHX_HTML"

$DOCKER run --rm \
    --volumes-from $CONTAINER \
    $IMAGE_ID/tester bash -c \
    "doxygen $BLDDIR/doc/doxygen/Doxyfile"

#
# Push documentation pages
#

git clone --depth 1 --no-checkout https://$GIT_OAUTH_TOKEN@github-fn.jpl.nasa.gov/isce-3/pr-docs
cd pr-docs
git reset

mkdir -p $ghprbPullId
cd $ghprbPullId

docker cp $CONTAINER:$BLDDIR/doc/. .
mv html/* .

git add .
git status

git config --local user.name  "gmanipon"
git config --local user.email "gmanipon@jpl.nasa.gov"
git commit -m "PR $ghprbPullId ($BUILD_URL)" && git push || echo "no changes committed"
