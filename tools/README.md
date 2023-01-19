## Docker CLI scripts

### Requirements

You must have these available on the host system:
* Python 3.7+
* Docker

To run CUDA tests, you will also need:
* NVIDIA Docker runtime
* CUDA Driver 396.26+

(These are just the requirements for the "host system" that runs the docker containers. ISCE3 itself has a much more extensive set of prerequisites, which will be automatically installed inside the docker containers.)

### Basic usage

```sh
./tools/run.py all
```

This will run the typical "all" sequence which is intended to mimic our CI build/testing pipeline. More options can be specified, e.g.

```sh
./tools/run.py -B build-alpine -i alpine setup
```

This will run the "setup" step for the "alpine" image set, using a build directory named "build-alpine". See the run.py script for a list of steps, and imagesets/imgset.py for their default implementations.

### Advanced usage

#### CI Debugging

CI failures should be reproducible locally via these scripts.
Running the "all" sequence should be sufficient.
```sh
./tools/run.py all
```

If additional info is needed, you can "drop in" to the builder image and run commands interactively. E.g.
```sh
./tools/run.py dropin
```

This will put you in the build directory with the same environment the tests ran in.
You can then run commands to isolate the error:
```sh
ctest -R my_failing_test --verbose
```

#### Local development

These scripts can be used as a local development environment,
to sidestep installing all the development libraries required to build ISCE3.
First build the docker images, run CMake configuration, and compile the code.
```sh
./tools/run.py setup configure build
```

After some local development, you can just rerun the build step - CMake will only recompile files which have been modified.
```sh
./tools/run.py build
```

#### Updating conda specfiles

See [Building identical conda environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#building-identical-conda-environments)

- Create a fresh conda environment
- Install runtime dependencies from requirements.txt
- `conda list --explicit > runtime/spec-file.txt`
- Install dev dependencies from requirements.txt
- `conda list --explicit > dev/spec-file.txt`
- Make sure the dev dependencies are a superset of the runtime dependencies
