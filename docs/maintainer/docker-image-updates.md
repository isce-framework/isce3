# Docker Image Updates

## Background

Docker images are the primary software deliverable of the NISAR Algorithm Development
Team (ADT) to the Science Data System (SDS). ADT images include the ISCE3 software,
containing NISAR science application workflows, as well as the NISAR Quality Assurance
(QA) and SoilMoisture software packages.

ADT Docker images use an Oracle Linux 8 base image with additional security patches
applied. [Miniforge](https://github.com/conda-forge/miniforge) is used to install
additional conda packages within the container image. In order to ensure that Docker
builds are reproducible, the ADT pins the base image and all installed packages such
that the version of each is explicitly specified and immutable.

A set of "spec files" (text files containing a list of concrete package specifications)
in the ISCE3 repository store the pinned version of each conda package in the ADT Docker
image. ISCE3's runtime and build-time dependencies are separated into distinct spec
files. The QA software dependencies are a subset of ISCE3's runtime dependencies, but
the SoilMoisture software has its own separate conda environment and spec file. These
spec files are tracked by Git and can be used to exactly reproduce each conda
environment on the same platform.

From time to time, it's desirable to upgrade the Docker base image and conda packages to
ensure that the ADT image includes the latest available software patches. Typically,
this procedure is carried out 2-3 weeks before a scheduled ADT delivery in order to
ensure there's sufficient time to test the environment changes.

## Procedure

### Updating the base Docker image

1. Find the latest available security-patched Oracle Linux image from the
[`docker-release-local/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux/`](https://artifactory.jpl.nasa.gov/ui/repos/tree/General/docker-release-local/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux)
repository on JPL Artifactory.

    !!! note
        Docker images in the repository are labeled by the OS major and minor version
        number and creation date in `YYMMDD` format (e.g. `8.10.250801`).

1. Get the image's SHA-256 digest. This can be found in the "Checksums" section of the
`manifest.json` (or `list.manifest.json`) file associated with the image in the
Artifactory repository. Alternatively, it can be found by pulling and inspecting the
image locally. For example, if the image tag is `8.10.250801`,

    ```shell
    $ REPO='cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux'
    $ TAG='8.10.250801'
    $ docker pull $REPO:$TAG
    $ docker inspect --format='{{index .RepoDigests 0}}' $REPO:$TAG | grep -oE 'sha256:[a-zA-Z0-9]+'
    ```

1. Modify `tools/imagesets/oracle8conda/runtime/Dockerfile` to replace the old base
image tag and digest with the new ones, e.g.

    ```diff title="Dockerfile"
    ARG repository=cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux
    -ARG tag=8.10.250701
    +ARG tag=8.10.250801
    -ARG digest=sha256:f0c2bab2b3d2b7483ab7dcd334a6230491ab8dece3c234221c1b30e5e23a4262
    +ARG digest=sha256:e0c73a62eccada432f9170b734bacb38e0be65fa1d3d660ee7255c56d8c16462
    FROM ${repository}:${tag}@${digest}

    ...
    ```

### Updating conda packages

The `tools/create_spec_file.py` script in the ISCE3 repository can be used to regenerate
the spec files that list the pinned versions of each conda package within the ADT Docker
image.

The script will spin up a Docker image, create a conda environment within the image
containing the required dependencies, and serialize the environment contents to a text
file within the ISCE3 repository.

!!! note
    The Docker base image and Miniforge version are hard-coded in the Python script. It
    may be necessary to manually update them (especially the Miniforge version) to match
    the versions used by the latest NISAR ADT Docker image.

The script should be run three times with different arguments to update each of the spec
files in the repository. Example usage from the root of the ISCE3 repository is shown
below.

```shell
$ ./tools/create_spec_file.py runtime
$ ./tools/create_spec_file.py dev
$ ./tools/create_spec_file.py soil-moisture
```
