
:orphan:

# Building and installing ISCE3


## Installing from conda-forge

If you just want to use ISCE3, you can avoid building it yourself by installing from our conda-forge package.

```bash
conda install -c conda-forge isce3
```


# Building with CMake

If you need to develop on ISCE3, you will need to build ISCE3 from source using pip.

```bash
git clone https://github.com/isce-framework/isce3
cd isce3
```

Make sure you have all the requirements for ISCE3 installed, and all the necessary packages available in your Python environment.
If you don't have such an environment already, we recommend using [miniforge](https://github.com/conda-forge/miniforge) to install all of these packages to a self-contained conda environment.

Create an environment for ISCE3 and activate it:

```bash
conda create -n isce3
conda activate isce3
```

Create a file called `requirements.txt` containing the following:

```
eigen
fftw
gdal
h5py
hdf5
numpy
pybind11
pyre
pysolid
python
ruamel_yaml
scipy
shapely
yamale
cmake
pkgconfig
```

Then run `conda install -y --file requirements.txt` to install them.

Now that you have all the prerequisites, it is time to run the build.
Pip should handle detecting your python installation and the prerequisites you just installed.

```bash
pip install .
```

If that command completes successfully, ISCE3 will now be available in your environment.

```
python3
>>> import isce3
```

## Building with CMake (Advanced)


If you need to specify additional build options that are not available through scikit-build-core, you can build using CMake directly.

```bash
mkdir build && cd build
cmake .. -GNinja -DCMAKE_INSTALL_PREFIX=./install
ninja install
```

After installing, make sure to run the unit tests to check that ISCE3 is behaving as expected:

```bash
ctest --output-on-failure
```

If many of these unit tests are failing, it is a good indicator that some python package is missing from your environment, or something has been configured incorrectly.

Once ISCE3 is installed and passing unit tests, making it available for import is more difficult than when using scikit-build-core.
You will need to add the built Python packages / extensions to your PYTHONPATH, and also add the ISCE3 C++ library to your library loader path.

Linux:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(realpath install/lib*)
export PYTHONPATH=$PYTHONPATH:$(realpath install/packages)
```

macOS:
```bash
export DYLD_FALLBACK_LIBRARY_PATH=$DYLD_FALLBACK_LIBRARY_PATH:$(realpath install/lib*)
export PYTHONPATH=$PYTHONPATH:$(realpath install/packages)
```

ISCE3 should now be available in your environment.

```
python3
>>> import isce3
```
