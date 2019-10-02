# Centos Base Images for ISCE v3

## nisar/base
This is the base image based on the `nvidia/cuda:9.2-runtime-centos7` image. It installs the runtime dependencies of ISCE v3 which includes python3 using miniconda.

1. Python requirements are captured in requirements.txt.base
2. User named "conda" is the default user in the container

### build
```
./build-base.sh <tag>
```

### run
```
docker run -u $ID:$(id -g) nisar/base:<tag>
```
or
```
docker run --runtime=nvidia -u $ID:$(id -g) nisar/base:<tag>
```
or
```
nvidia-docker run -u $ID:$(id -g) nisar/base:<tag>
```

## nisar/dev
This is the development image based on the `nisar/base` image. It installs the development libraries and dependencies necessary for development of ISCE v3.

1. Python requirements are captured in requirements.txt.dev
2. User named "conda" is the default user in the container

### build
```
./build-dev.sh <tag>
```

### run
```
docker run -u $ID:$(id -g) nisar/dev:<tag>
```
or
```
docker run --runtime=nvidia -u $ID:$(id -g) nisar/dev:<tag>
```
or
```
nvidia-docker run -u $ID:$(id -g) nisar/dev:<tag>
```

## nisar/isce-ops
This is the runtime ISCE v3 image based on the `nisar/base` image. It uses a multi-stage docker build to compile, install and RPM-ize ISCE v3 in a `nisar/dev` container then copies the ISCE v3 RPM into the `nisar/base` image to install the ISCE v3 RPM.

1. User named "conda" is the default user in the container
2. ISCE v3 installed at /opt/isce

### build
```
TAG=$(date -u +%Y%m%d)
WORKSPACE=<path to cloned isce repo>
./build-isce-ops.sh $TAG $WORKSPACE
```

### run
```
docker run -u $ID:$(id -g) nisar/isce-ops:<tag>
```
or
```
docker run --runtime=nvidia -u $ID:$(id -g) nisar/isce-ops:<tag>
```
or
```
nvidia-docker run -u $ID:$(id -g) nisar/isce-ops:<tag>
```

