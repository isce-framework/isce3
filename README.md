ISCE - InSAR Scientific Computing Environment
=============================================

[![Build Status](
https://github.com/isce-framework/isce3/actions/workflows/build-and-run.yml/badge.svg)](
https://github.com/isce-framework/isce3/actions/workflows/build-and-run.yml)

The InSAR Scientific Computing Environment (ISCE) is an open source library for
processing spaceborne and airborne Interferometric Synthetic Aperture Radar
(InSAR) data.

This project is a successor to the [ISCE2](https://github.com/isce-framework/isce2)
framework. It is a ground-up redesign focusing on improved modularity,
documentation, and test-driven development.

Its initial development was funded by NASA's Earth Science Technology Office
(ESTO) under the Advanced Information Systems Technology (AIST) 2008 and is
currently being funded under the NASA-ISRO SAR (NISAR) project.

- [Installation](#installation)
- [Getting Started](#getting-started)
- [Contributing](#contributing)
- [License](#license)

> :warning: **NOTICE**: ISCE3 is in early development - its
features and interface are subject to change

## Installation

ISCE3 is available as a conda package via conda-forge. It has CPU-only and GPU-enabled versions available. You can install the CPU-only version with the following command:

```bash
conda install -c conda-forge isce3-cpu
```

Alternatively, if you have a NVIDIA GPU with CUDA support and the CUDA driver installed, you can use the GPU-enabled package for faster processing:

```bash
conda install -c conda-forge isce3-cuda
```

If you need to build ISCE3 from source, please refer to our more in-depth [build instructions](https://isce-framework.github.io/isce3/buildinstall/).

## Getting Started

See the documentation at https://isce-framework.github.io/isce3/

## Contributing

Contributions are welcome. Refer to the [contributing guide](CONTRIBUTING.md)
for tips on getting started and guidelines for contributions. Please let us know
if you encounter a bug by
[filing an issue](https://github.com/isce-framework/isce3/issues)

## License

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS.
USE AT YOUR OWN RISK.

USE OF THIS SOFTWARE IS CONTROLLED BY THE TERMS SPECIFIED IN THE LICENSE FILE.
EXPORT OF THIS SOFTWARE IS SUBJECT TO U.S. EXPORT CONTROL LAWS AND REGULATIONS
AS SPECIFIED BY THE TERMS OF ITS EAR99 NLR CLASSIFICATION.

PLEASE READ THE LICENSE FILE FOUND IN THIS PACKAGE FOR FURTHER DETAILS.
