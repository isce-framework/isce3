# Contributing Guide

There are many ways to contribute to the isce3 project, including:

- Submitting bug reports and feature requests
- Sharing tutorials or jupyter-notebooks
- Improving documentation
- Fixing bugs, typos, and open issues
- Reviewing open pull requests
- Developing new features

This document discusses tips for working on the isce3 code base and guidelines
for contributing to the project.

## Table of contents

- [Development process](#development-process)
    - [Forking the isce3 repository](#forking-the-isce3-repository)
    - [Setting up a development environment](#setting-up-a-development-environment)
    - [Developing your contribution](#developing-your-contribution)
        - [Incorporating upstream changes](#incorporating-upstream-changes)
        - [Documentation and tests](#documentation-and-tests)
        - [Style guidelines](#style-guidelines)
    - [Submitting a pull request](#submitting-a-pull-request)
    - [Review process](#review-process)

## Development process

### Forking the isce3 repository

If you are a first-time contributor, the first thing you should do is fork the
isce3 repository to create your own copy of the project. (External contributors
cannot directly create or modify branches on the isce3 repository.)

1. Go to https://github.com/isce-framework/isce3 and click on the "Fork"
   button.
1. Clone the forked project to your local computer:

    ```sh
    $ git clone https://github.com/my-user-name/isce3
    ```

1. Add the `upstream` remote repository. This allows you to track "upstream"
   changes to the main repository:

    ```sh
    $ cd isce
    $ git remote add upstream https://github.com/isce-framework/isce3.git
    ```

Now, `git remote -v` should show two remote repositories: `upstream`, which
refers to the main isce3 repository, and `origin`, which refers to your personal
fork.

```
$ git remote -v
origin          https://github.com/my-user-name/isce3.git (fetch)
origin          https://github.com/my-user-name/isce3.git (push)
upstream        https://github.com/isce-framework/isce3.git (fetch)
upstream        https://github.com/isce-framework/isce3.git (push)
```

### Setting up a development environment

Proceed following the
[Linux](https://isce-framework.github.io/isce3/install_linux.html) or
[macOS](https://isce-framework.github.io/isce3/install_osx.html)
instructions for building from source, using your forked copy of the repository
instead of the main repo.

### Developing your contribution

1. Pull the latest changes from upstream:

    ```sh
    $ git checkout develop
    $ git fetch upstream
    $ git merge upstream/develop --ff-only
    ```

1. Create a new branch for the feature you want to work on. Choose a meaningful
   name for your branch that briefly describes the intended changes:

    ```sh
    $ git checkout -b my-branch-name
    ```

1. Begin working on your changes. Be sure to commit your changes locally as you
   progress (with `git add` and `git commit`), using a
   [properly-formatted](https://gist.github.com/robertpainsi/b632364184e70900af4ab688decf6f53)
   commit message.

#### Incorporating upstream changes

Periodically, it may be necessary to merge in changes from the upstream
development branch to your local feature branch in order to get new bugfixes or
important features. The preferred method of doing this is via a "fast-forward",
which avoids creating a merge commit:

```sh
$ git checkout my-branch-name
$ git fetch upstream
$ git merge upstream/develop --ff-only
```

In some cases, it may not be possible for your changes to be resolved with the
upstream history by a simple fast-forward. If the fast-forward fails, then fall
back to the standard merge mode:

```sh
$ git checkout my-branch-name
$ git fetch upstream
$ git merge upstream/develop
```

It may be necessary to resolve merge conflicts manually in this case. All
conflicts with the `upstream/develop` branch must be resolved before the changes
can be tested by the automated continuous integration (CI) system or merged into
the upstream repository.

#### Documentation and tests

- All code should be documented.
- All code should have unit tests.
- Before issuing or updating a pull request (PR), run the tests locally to make
  sure they succeed.

#### Style guidelines

- Ensure that all C++ and CUDA code conforms to the isce3 style guide by running
  [clang-format](https://clang.llvm.org/docs/ClangFormat.html) on your changes
  before pushing.
- All Python code should follow
  [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidance.

### Submitting a pull request

1. Push your changes back to your fork on GitHub, creating or updating the remote
   branch with the same name:

    ```sh
    $ git checkout my-branch-name
    $ git push origin my-branch-name
    ```

1. Go to the GitHub page for your branch
   (https://github.com/isce-framework/isce3/tree/my-branch-name) and
   click on the "New pull request" button. Write a clear and concise title for
   your PR and include an explanation of the proposed changes before submitting.

You should now be able to view your submission in the list of
[open PRs](https://github.com/isce-framework/isce3/pulls)

Subsequent `git push`es from your feature branch will automatically update the
PR with new changes.

### Review process

Every developer working on the project has their code reviewed. The process is
intended to be a friendly conversation from which we all learn and improve the
quality of the project.

- Review may be requested by a PR author and/or other team members. Reviewers
  can make comments, request changes, or approve the PR, indicating that it has
  been carefully examined and is ready for merging.
- Before a PR can be merged, it must be approved by at least two core team
  members.

### Testing and scanning

- CI jobs that build and test the code are automatically triggered upon each PR
  update. The CI tests must pass before your PR can be merged. To avoid overuse
  of these resources, it's helpful to test all changes locally before committing
  and push commits in batches rather than individually.
- Each commit is scanned using
  [detect-secrets](https://github.com/Yelp/detect-secrets) and
  [CodeQL](https://codeql.github.com/) to identify possible security
  vulnerabilities and exposed credentials. Commits containing either may be
  blocked.
- Linters may be run on the changes and may reject PRs that don't conform to the
  project's style conventions.

After all required checks have passed, the PR may be merged by a maintainer.
