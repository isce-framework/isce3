================================================================================
ISCE - Insar Scientific Computing Environment
================================================================================

This is the Interferometric synthetic aperture radar Scientific Computing
Environment (ISCE).  Its initial development was funded by NASA's Earth Science
Technology Office (ESTO) under the Advanced Information Systems Technology
(AIST) 2008 and is currently being funded under the NASA-ISRO SAR (NISAR)
project.

THIS IS RESEARCH CODE PROVIDED TO YOU "AS IS" WITH NO WARRANTIES OF CORRECTNESS.
USE AT YOUR OWN RISK.

Use of this software is controlled by a non-commercial use license agreement
provided by the California Institute of Technology Jet Propulsion Laboratory.
You must obtain a license in order to use this software.  Please consult the
LICENSE file found in this package.

ISCE is a framework designed for the purpose of processing Interferometric
Synthetic Aperture Radar (InSAR) data.  The framework aspects of it have been
designed as a general software development framework.  It may have additional
utility in a general sense for building other types of software packages.  In
its InSAR aspect ISCE supports data from many space-borne satellites and one
air-borne platform.  We continue to increase the number of sensors supported.
At this time the sensors that are supported are the following: ALOS, ALOS2,
COSMO_SKYMED, ENVISAT, ERS, KOMPSAT5, RADARSAT1, RADARSAT2, RISAT1, Sentinel1,
TERRASARX, and UAVSAR.

================================================================================
Contents
================================================================================

1.  Software Dependencies
1.1 Installing pyre and config
1.2 Installing software dependencies with standard package managers
1.3 Installing Virtual Machine Images with Dependencies Pre-Installed
1.4 Installing dependencies with provided setup script
1.5 Hints for installing dependencies by hand.
1.6 Note On 'python3' Exectuable Convention
2.  Building ISCE
2.1 Working with config's mm
2.2 Install ISCE
2.3 Setup Your Environment
3.  Running ISCE
3.1 Running ISCE from the command line
3.2 Running ISCE in the Python interpreter
3.3 Running ISCE with steps
3.4 NOTE on DEM
4.  Input Files
5.  Component Configurability
5.1 Component Names: Family and Instance
5.2 Component Configuration Files: Locations, Names, Priorities
5.3 Component Configuration Help

================================================================================
1. Software Dependencies
================================================================================

Basic:
------
pyre-1.0
config
gcc >= 4.7
fftw 3.2.2
Python >= 3.3
curl - for automatic DEM downloads

For a few sensor types:
-----------------------
gdal python bindings >= 2.0 - for RadarSAT2
hdf5 >= 1.8.5 and h5py >= 1.3.1  - for COSMO-SkyMed
spiceypy  - for RadarSAT1

For mdx (image visualization tool):
-----------------------------------
Motif libraries and include files
ImageMagick - for mdx production of kml file (advanced feature)
grace - for mdx production of color table and line plots (advanced feature)

For the "unwrap 2 stage" option:
--------------------------------
RelaxIV and Pulp are required.  Information on getting these packages if
you want to try the unwrap 2 stage option:
* RelaxIV (a minimum cost flow relaxation algorithm coded in C++ by
Antonio Frangioni and Claudio Gentile at the University of Pisa,
based on the Fortran code developed by by Dimitri Bertsekas while
at MIT) available by request at http://www.di.unipi.it/~frangio.
So that ISCE will compile it properly, the RelaxIV files should
be placed in the directory: 'contrib/UnwrapComp/src/RelaxIV'.
* PULP: Use easy_install or pip to install it or else clone it from,
https://github.com/coin-or/pulp.  Make sure the path to the installed
pulp.py is on your PYTHONPATH environment variable (it should be the case
if you use easy_install or pip).

Optional for splitSpectrum, GPUtopozero, and GPUgeo2rdr
cython3 - must have an executable named cython3 (use a symbolic link)
cuda - for GPUtopozero and GPUgeo2rdr

--------------------------------------------------------------------------------
1.1 Installing pyre and config
--------------------------------------------------------------------------------

Detailed instructions for installing pyre and config are found at:

pyre.orthologue.com/install

--------------------------------------------------------------------------------
1.2 Installing software dependencies with standard package managers
--------------------------------------------------------------------------------

The easiest way to install most of these is with package managers such as
'apt-get' on Linux systems or 'macports' on MacOsX or anaconda.  To use these,
however, may require that you have superuser permission on your computer. The
following URL gives additional information on installing prerequisites for
ISCE:

https://winsar.unavco.org/portal/wiki/Manual%20installation%20using%20repository%\
20managers/

If it is not possible for you to install the software yourself and you
can't convince the System Administrator on your computer to install the
dependencies, then we provide virtual machine images (VMs) with the
dependencies pre-installed (see Section 1.2).

For the truly adventurous who want to install dependencies by hand we provide
some hints in Section 1.5.

When you have installed the dependencies you can skip the other sections about
installing the dependencies and read Section 1.6 about the 'python3' convention
and then Section 2 on building ISCE and configuring your environment.

--------------------------------------------------------------------------------
1.3 Installing Virtual Machine Images with Dependencies Pre-Installed
--------------------------------------------------------------------------------

If you don't have superuser privileges on your machine and your system is not
up to date with the software dependencies required to use ISCE, then you can
download Virtual Machine Images (VMs) at the following URL:

Full link: http://earthdef.caltech.edu/boards/4/topics/305
Simple link: http://tinyurl.com/iscevm

Instructions on how to install the Virtual Machines are given there.

--------------------------------------------------------------------------------
1.4 Installing dependencies with provided setup script
--------------------------------------------------------------------------------

This distribution includes a very **experimental** script that is designed to
download, build, and install all relevant packages needed for ISCE (except for
h5py, which presently must be built by hand but is only needed for Cosmo-Skymed,
spiceypy, only needed for RadarSAT1, and gdal python bindings).  This script is
meant as a last resort for those adventurous persons who may not have root
privileges on their machine to install software with standard package managers
or a virutal machine (VM) image (see Section 1.2 or 1.3).

The script is in the setup directory, and is called install.sh.  To run it, you
should cd to the setup directory, then issue the command

> install.sh -h

to see instructions on how to run the script.  The minimal command option is
simply,

> install.sh -p <INSTALL_PATH>

where <INSTALL_PATH> is a path where you have permission to create files.  This
will check whether the dependencies exist on your default paths and then install
those that do not on the specified path.

The path should be in a local directory away from the system areas to avoid
conflicts and so that administrator privileges are not needed. The config.py
file contains a list of urls where the packages are currently downloaded from
Commenting out a particular package will prevent installation of that package.
If the specified server for a particular package in this file is not available,
then you can simply browse the web for a different server for this package and
replace it in the config.py file. Below under the "Building ISCE" section,
there are instructions on how to point to these packages for building ISCE.

Once all these packages are built, you must setup your PATH and LD_LIBRARY_PATH
variables in the unix shell to ensure that these packages are used for compiling
and linking rather than the default system packages.

--------------------------------------------------------------------------------
1.5 Hints for installing dependencies by hand.
--------------------------------------------------------------------------------

If you would prefer to install all these packages by hand, follow this procedure:

Compile the following

for Radarsat2, Sentinel1A and Tandem-X gdal with python bindings >= 1.9

Building gcc/gfortran
---------------------
Building gcc from source code can be a difficult undertaking.  Refer to the
detailed directions at  http://gcc.gnu.org/install/ for further help.

Building fftw-3.2.2
-------------------
Get fftw-3.2.2 from: http://www.fftw.org/fftw-3.2.2.tar.gz
Untar the file fftw-3.2.2.tar.gz using
tar -zxvf fftw-3.2.2.tar.gz
cd fftw-3.2.2
then run ./configure --enable-single --enable-shared --prefix=<directory>
where <directory> is the full path to an installation location where you have
write access. Then run,

make
make install

Building Python
---------------
Get the Python source code from http://www.python.org/ftp/python/3.3.5/Python-3.3.5.t\
gz

Untar the file Python-3.3.5.tgz using

tar -zxvf Python-3.3.5.tgz
cd Python-3.3.5

Then run

./configure --prefix=<directory>

where <directory> is the full path to an installation location where you
have write access.  Then run,

make
make install

Building hdf5 [Only necessary for COSMO-SkyMed support]
-------------
Get the source code from:
http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.6.tar.gz

Building h5py [Only necessary for COSMO-SkyMed support]
Get the h5py source code from:
http://code.google.com/p/h5py/downloads/detail?name=h5py-1.3.1.tar.gz


Building gdal-bindings [Only necessary for Radarsat2, Tandem-X and Sentinel 1A]
----------------
On most linux distributions, gdal can installed along with its python bindings
using standard repository management tools.

If you don't have gdal, you can find instructions on building GDAL here
http://trac.osgeo.org/gdal/wiki/BuildHints

Remember to use configure with --with-python option to automatically build
python bindings.

Else, if you already have gdal installed on your system but without python
buindings, use easy_install corresponding to your python 3 version. You may
need to setup LD_LIBRARY_PATH correctly to include path to libgdal.so

easy_install GDAL


Building SpiceyPy  [Only necessary for Radarsat1]
----------------
JPL's CSPICE library (http://naif.jpl.nasa.gov/naif/toolkit_C.html) is needed
for this. Follow instructions at https://github.com/Apollo117/SpiceyPy to
install SpiceyPy, after installing CSPICE.

Once all these packages are built, you must setup your PATH and LD_LIBRARY_PATH
variables in the unix shell to ensure that these packages are used for compiling
and linking rather than the default system packages.


To use the Spice software you will need to download the data files indicated in
the component/isceobj/Orbit/db/kernels.list file.  You should download those
files into that directory (or else make soft links in that directory to where
you download them) so that ISCE can find them in the place it expects.

--------------------------------------------------------------------------------
1.6 Note On 'python3' Executable Convention
--------------------------------------------------------------------------------

We follow the convention of most package managers in using the executable
'python3' for Python3.x and 'python' for Python2.x.  This makes it easy to turn
Python code into executable commands that know which version of Python they
should invoke by naming the appropriate version at the top of the executable
file (as in #!/usr/bin/env python3 or #!/usr/bin/env python).  Unfortunately,
not all package managers (such as macports) follow this convention.  Therefore,
if you use one of a package manager that does not create the 'python3'
executable automatically, then you should place a soft link on your path to
have the command 'python3' on your path.  Then you will be able to execute an
ISCE application such as 'insarApp.py as "> insarApp.py" rather than as
"> /path-to-Python3/python insarApp.py".

================================================================================
2. Building ISCE
================================================================================

--------------------------------------------------------------------------------
2.1 Working with config's mm
--------------------------------------------------------------------------------

The config package builder uses the command "mm" to build and install the pyre
and isce codes. To add non-standard paths to mm, such as the location of pyre,
run the command 'mm.paths' while in the directory to be added.  For instance,
pyre-1.0 is a dependency whose path must be know to mm before building isce.
Go to the pyre-1.0 top directory and then issue the command, "mm.paths" at the
command line.  That is all that is necessary to inform mm of the path.

--------------------------------------------------------------------------------
2.2 Installing ISCE with mm
--------------------------------------------------------------------------------

To install ISCE, simply enter the top level of the ISCE directory and issue the
command "mm".  The directory products will contain the installed libraries,
headers, and Python applications and components.

--------------------------------------------------------------------------------
2.3 Set up your environment
--------------------------------------------------------------------------------

Add the isce "products/packages" directory to your PYTHONPATH.

==============================================================================
END OF FILE
==============================================================================
