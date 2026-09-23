# Security Scanning

## Docker image scanning with grype

The NISAR ADT uses [grype](https://github.com/anchore/grype) to scan ADT images for
Common Vulnerabilities and Exposures (CVEs). The tool scans for software packages in a
Docker container image and compares against a database of known security
vulnerabilities. The database is updated periodically, so scanning the same image on
different dates may produce different results.

Grype is typically run manually prior to each ADT delivery using the release image.

### Installing grype

Grype can be installed via an installation script or one of several package managers
using the instructions found
[here](https://github.com/anchore/grype?tab=readme-ov-file#installation).

It's also available as a conda package from conda-forge.

```shell
$ conda install -c conda-forge grype
```

### Running grype

Grype can be run by passing the Docker image tag or ID to the `grype` command,

```shell
$ grype $DOCKER_TAG
```

where `$DOCKER_TAG` is the full tag or ID of the Docker image.

For example, for the NISAR SDS R05.00.0 release, run

```shell
$ grype nisar-adt/isce3:r05.00.0-v0.25.1
```

Grype can also produce more detailed output in JSON format using the `--output=json`
flag,

```shell
$ grype --output=json $DOCKER_TAG > $OUTPUT_JSON
```

where `$OUTPUT_JSON` is the name of the desired output JSON file.

## Code scanning with detect-secrets

[`detect-secrets`](https://github.com/Yelp/detect-secrets) is used to scan the codebase
for exposed secrets such as credentials using regex-based rules, entropy detectors, and
keyword detectors.

The tool is run automatically via [pre-commit.ci](https://pre-commit.ci/) as a
Continuous Integration (CI) job on each ISCE3 Pull Request (PR).
If secrets are found, they should be removed prior to merging the PR.
False alarms can be added to the [secrets
baseline](https://github.com/Yelp/detect-secrets?tab=readme-ov-file#adding-new-secrets-to-baseline).

Security scanning using `detect-secrets` can also be performed manually using pre-commit
locally.

### Installing pre-commit

pre-commit can be installed via pip using the instructions found
[here](https://pre-commit.com/#installation).

It's also available as a conda package from conda-forge.

```shell
$ conda install -c conda-forge pre-commit
```

### Running the detect-secrets hook

Run the `detect-secrets` hook on all source files via pre-commit (should be done from
the root of the ISCE3 repository).

```shell
$ pre-commit run --all-files detect-secrets
```
