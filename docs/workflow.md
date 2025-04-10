# Workflow testing

Workflow tests exercise full NISAR SAS workflows on large input test datasets in order to validate the deliverable docker image for each ADT release. In addition, (some subset of) workflow tests are generally delivered to the PGE team for their own testing. Unlike unit tests, workflow tests require large binary files that are stored separately from the isce3 code repository.

This document describes how workflow tests are organized and executed in isce3.


## Organization

Workflow testing infrastructure in isce3 is integrated with the Docker run scripts used for Continuous Integration (CI) automation and manual testing of ADT deliverables, located in the ["tools"](https://github-fn.jpl.nasa.gov/isce-3/isce/tree/develop/tools) subdirectory of the isce3 repository.

### Runconfigs

Each workflow test case is defined by a single runconfig YAML file which provides the input parameters to the SAS workflow. Runconfig files for each test are version-controlled as part of the isce3 repository. They can be found in:

[isce/tools/imagesets/runconfigs/](https://github-fn.jpl.nasa.gov/isce-3/isce/tree/develop/tools/imagesets/runconfigs)

### Datasets

Workflow test datasets include all of the large binary input files needed to run a particular workflow, accompanied by a plain text README file. Currently, each dataset used for ADT testing is stored on artifactory here:

https://artifactory.jpl.nasa.gov/artifactory/general-develop/gov/nasa/jpl/nisar/adt/data/

Test datasets are fetched at runtime by isce3's Docker run scripts for testing.

Each dataset may be used for one or more workflow tests, and some workflow tests may require more than one input dataset. The [workflowdata.py](https://github-fn.jpl.nasa.gov/isce-3/isce/blob/develop/tools/imagesets/workflowdata.py) script defines the mapping between each test case and the input dataset(s) that it requires.


## Running workflow tests

Workflow tests can be executed using isce3's Docker CLI tool via the [`tools/run.py`](https://github-fn.jpl.nasa.gov/isce-3/isce/blob/develop/tools/run.py) script. A description of the requirements and general usage of this script can be found [here](https://github-fn.jpl.nasa.gov/isce-3/isce/blob/develop/tools/README.md).

Prior to running workflow tests, you will need a [classic Git OAUTH Token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic) with `repo` permissions. The string name of the token should be saved to the `GIT_OAUTH_TOKEN` environment variable on your system.

The basic steps for executing workflow tests are as follows:

1. First, generate an isce3 redistributable image using the Docker CLI tool

   ```
   $ tools/run.py setup configure build test makepkg makedistrib makedistrib_nisar
   ```

1. Next, fetch test datasets from artifactory

   **WARNING**: Fetching all test datasets may take a long time and requires significant disk utilization. There is currently no support for fetching individual test datasets via the `tools/run.py` script and no automatic local caching.

   ```
   $ tools/run.py fetchdata
   ```

1. Finally, run each workflow test

   **NOTE**: Each workflow test suite below is independent and can be run individually or out-of-order.

   ```
   $ tools/run.py rslctest gslctest gcovtest insartest end2endtest
   ```
