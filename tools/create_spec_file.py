#!/usr/bin/env python

from __future__ import annotations

import argparse
import os
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from textwrap import dedent


def build_img(
    steps: str,
    context: os.PathLike | str = ".",
    tag: str | None = None,
    target: str | None = None,
) -> str:
    """
    Build a Docker image.

    Parameters
    ----------
    steps : str
        A string in Dockerfile syntax containing build instructions.
    context : str or path-like, optional
        The build context. Should be a URL or path to a directory. Defaults to the
        current working directory.
    tag : str or None, optional
        An optional name/tag to apply to the resulting Docker image. If None, no tag is
        applied. Defaults to None.
    target : str or None, optional
        The target build stage, for use with multi-stage builds. If not None, the
        builder skips commands after the target stage. Defaults to None.

    Returns
    -------
    str
        The image ID.
    """
    # Build a list of command-line arguments to run as a subprocess. The Dockerfile
    # contents will be passed via stdin.
    args = ["docker", "build", "--network=host", "--quiet", "--file=-"]
    if tag is not None:
        args.append(f"--tag={tag}")
    if target is not None:
        args.append(f"--target={target}")
    args.append(os.fsdecode(context))

    res = subprocess.run(args, capture_output=True, check=True, text=True, input=steps)
    return res.stdout.strip()


def remove_img(img: str) -> None:
    """
    Remove (and un-tag) a Docker image.

    Parameters
    ----------
    img : str
        The image ID or tag. If a tag is supplied, if the image has multiple tags, only
        the tag is removed. Otherwise, both the image and tag are removed.
    """
    subprocess.run(["docker", "rmi", img], capture_output=True, check=True)


@contextmanager
def build_temp_img(*args, **kwargs) -> Generator[str, None, None]:
    """
    Build a temporary Docker image and remove it upon exiting the context manager.

    Parameters
    ----------
    *args, **kwargs
        Positional and keyword arguments to pass to `build_img()`.
    """
    img = build_img(*args, **kwargs)
    try:
        yield img
    finally:
        remove_img(img)


def create_pkg_list(img: str, env: str = "base") -> str:
    """
    List conda packages in a Docker image.

    Parameters
    ----------
    img : str
        The Docker image ID or tag.
    env : str, optional
        The name of the conda environment. Defaults to 'base'.

    Returns
    -------
    str
        The conda environment specs.
    """
    conda_cmd = f"conda list --name={env} --explicit"
    args = ["docker", "run", "--rm", img, "bash", "-c", conda_cmd]
    res = subprocess.run(args, capture_output=True, check=True, text=True)
    return res.stdout


def sort_pkgs(specs: str) -> str:
    """
    Sort the contents of a conda spec file alphabetically.

    Parameters
    ----------
    specs : str
        A multi-line string listing the specs of a conda environment, such as produced
        by `conda list`. Any initial lines that begin with '#' or '@' are considered
        header lines and their order is preserved.

    Returns
    -------
    str
        A copy of `specs` with the contents sorted alphabetically.
    """
    lines = specs.splitlines()

    # Get the index of the first line that doesn't begin with '#' or '@'.
    index = 0
    for index, line in enumerate(lines):
        if not line.startswith(("#", "@")):
            break

    sorted_lines = lines[:index] + sorted(lines[index:])
    return "\n".join(sorted_lines) + "\n"


def create_spec_file(target: str) -> None:
    """
    Create a new conda environment spec file.

    Parameters
    ----------
    target : {'runtime', 'dev', 'soil-moisture'}
        'runtime':
          Create a spec file containing runtime dependencies of ISCE3.

        'dev':
          Create a spec file containing build-time dependencies of ISCE3.

        'soil-moisture':
          Create a spec file containing dependencies of the NISAR SWNG SoilMoisture
          software.
    """
    # The base Docker image to inherit from. This should match the base image used by
    # the NISAR ADT Docker images.
    base_repository = "cae-artifactory.jpl.nasa.gov:16003/gov/nasa/jpl/iems/sds/infrastructure/base/jplsds-oraclelinux"
    base_tag = "8.10.250401"
    base_digest = "sha256:e37c210f26cfe7660808e536937fcf6ac359b3ec5713211e46519d7a35973a57"
    base_img = f"{base_repository}:{base_tag}@{base_digest}"

    # The URL of the Miniforge installer script to run. This should match the Miniforge
    # version used by the NISAR ADT Docker image.
    miniforge_release = "24.11.3-2"
    miniforge_url = f"https://github.com/conda-forge/miniforge/releases/download/{miniforge_release}/Miniforge3-Linux-x86_64.sh"

    # Create a multi-stage Dockerfile. The 'base' stage installs conda within the image.
    # The 'runtime' stage inherits from 'base' and installs runtime dependencies in the
    # base environment. The 'dev' stage inherits from 'runtime' and installs build-time
    # dependencies in the same environment. The 'soil-moisture' stage inherits directly
    # from 'base' and installs the Soil Moisture dependencies in their own environment.
    dockerfile = dedent(
        f"""\
        FROM {base_img} AS base

        ENV CONDA_PREFIX=/opt/conda
        RUN curl -sSL {miniforge_url} -o Miniforge3.sh \
            && bash Miniforge3.sh -b -p $CONDA_PREFIX \
            && rm Miniforge3.sh
        ENV PATH="$CONDA_PREFIX/bin:$PATH"
        RUN conda config --set solver libmamba

        FROM base AS runtime

        COPY runtime/environment.yml /tmp/environment.yml
        RUN conda env update --name=base --file=/tmp/environment.yml

        FROM runtime AS dev

        COPY dev/environment.yml /tmp/environment.yml
        RUN conda env update --name=base --file=/tmp/environment.yml

        FROM base AS soil-moisture

        COPY distrib_nisar/requirements.soil-moisture.yml /tmp/environment.yml
        RUN conda env create --name=soil-moisture --file=/tmp/environment.yml
        """
    )

    # The path to the directory to be used as a Docker build context.
    context = Path(__file__).parent / "imagesets/oracle8conda"

    # Infer the name of the conda environment within the Docker image and the output
    # spec file path based on `target`.
    if (target == "runtime") or (target == "dev"):
        env = "base"
        spec_file = context / f"{target}/spec-file.txt"
    elif target == "soil-moisture":
        env = "soil-moisture"
        spec_file = context / "distrib_nisar/soilmoisture-spec-file.txt"
    else:
        raise ValueError(f"unexpected target: {target!r}")

    # Create a Docker image containing a conda environment with the required
    # dependencies. Using the Docker container, export a reproducible list of packages.
    # The Docker image is cleaned up automatically afterwards.
    with build_temp_img(dockerfile, target=target, context=context) as img:
        specs = create_pkg_list(img, env=env)

    # `conda list` exports packages in seemingly random order. Sort packages in
    # alphabetical order for cleaner diffs.
    sorted_specs = sort_pkgs(specs)

    # Write to the output spec file. If the file exists, it will be overwritten.
    spec_file.write_text(sorted_specs)


def main(args: list[str] | None = None) -> None:
    """
    Main command line entrypoint.

    Parameters
    ----------
    args : list of str or None, optional
        The list of arguments. If None, the argument list is taken from `sys.argv`.
        Defaults to None.
    """
    description = dedent(
        """
        Generate conda spec files for use with NISAR ADT Docker images.

        Reads a YAML file containing a set of conda environment specs, and creates a
        temporary Docker image containing a conda environment with those specs. Then,
        serializes the list of packages within the environment to a text file (a
        'spec file') that can be used to exactly reproduce the environment on the same
        platform.

        Three different specfiles can be created: the 'runtime' spec file contains all
        runtime dependencies (including CUDA dependencies) of ISCE3, the 'dev' spec file
        contains all packages from the 'runtime' spec file as well as all build-time
        dependencies of ISCE3, and the 'soil-moisture' spec file contains the runtime and
        build-time dependencies of the NISAR SNWG SoilMoisture software
        (https://github-fn.jpl.nasa.gov/NISAR-ADT/SoilMoisture).
        """
    )
    # Setup the argument parser.
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        choices=("runtime", "dev", "soil-moisture"),
        help="Which spec file to create",
    )

    # Parse the arguments and convert the result to a dict of keyword arguments to pass
    # to `create_spec_file()`.
    kwargs = vars(parser.parse_args(args))

    create_spec_file(**kwargs)


if __name__ == "__main__":
    main()
