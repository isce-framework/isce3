# ISCE3

[ISCE3](https://github.com/isce-framework/isce3) is the InSAR Scientific Computing Environment, a ground-up redesign of [ISCE2](https://github.com/isce-framework/isce2).

ISCE3 is an open-source library for spaceborne and airborne Synthetic Aperture Radar (SAR) data processing, released under the Apache 2.0 license.
It has been developed as the platform on which data processing workflows for the [NASA-ISRO SAR (NISAR)](https://nisar.jpl.nasa.gov/) mission are built. As a library, it provides a variety of data structures that are used frequently for InSAR data processing.

Although ISCE3's development has been primarily focused on supporting the NISAR mission, it has been designed to function as a general-purpose SAR processing framework. High-performance C++/CUDA algorithms are exposed via Python bindings for use in a variety of SAR processing scenarios.

## Building and installing ISCE3

ISCE3 is available as a conda package via conda-forge. It has CPU-only and GPU-enabled versions available. You can install the CPU-only version with the following command:

```bash
conda install -c conda-forge isce3-cpu
```

Alternatively, if you have a NVIDIA GPU with CUDA support and the CUDA driver installed, you can use the GPU-enabled package for faster processing:

```bash
conda install -c conda-forge isce3-cuda
```

If you need to build ISCE3 from source, please refer to our more in-depth [build instructions](buildinstall.md).
