ISCE - Insar Scientific Computing Environment
=============================================

[![Build Status](https://nisar-ci.jpl.nasa.gov/buildStatus/icon?job=isce-develop)](https://nisar-ci.jpl.nasa.gov/job/isce-develop/)|[![Nightly Build Status](https://nisar-ci.jpl.nasa.gov/buildStatus/icon?job=isce-develop-nightly-build)](https://nisar-ci.jpl.nasa.gov/job/isce-develop-nightly-build/)

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


Contents
========

* 1.0 Software Dependencies
* 1.1 Installing pyre and config
* 1.2 Installing software dependencies with standard package managers
* 1.3 Installing Virtual Machine Images with Dependencies Pre-Installed
* 1.4 Installing dependencies with provided setup script
* 1.5 Hints for installing dependencies by hand.
* 1.6 Note On 'python3' Exectuable Convention
* 2.0 Building ISCE
* 2.1 Working with config's mm
* 2.2 Install ISCE
* 2.3 Setup Your Environment
* 3.0 Running ISCE
* 3.1 Running ISCE from the command line
* 3.2 Running ISCE in the Python interpreter
* 3.3 Running ISCE with steps
* 3.4 NOTE on DEM
* 4.0 Input Files
* 5.0 Component Configurability
* 5.1 Component Names: Family and Instance
* 5.2 Component Configuration Files: Locations, Names, Priorities
* 5.3 Component Configuration Help


1.0 Software Dependencies
=========================

Basic:
------
* pyre-1.0
* config
* gcc >= 4.7
* fftw 3.2.2
* Python >= 3.3
* curl - for automatic DEM downloads

For a few sensor types:
-----------------------
* gdal python bindings >= 2.0 - for RadarSAT2
* hdf5 >= 1.8.5 and h5py >= 1.3.1  - for COSMO-SkyMed
* spiceypy  - for RadarSAT1

For mdx (image visualization tool):
-----------------------------------
* Motif libraries and include files
* ImageMagick - for mdx production of kml file (advanced feature)
* grace - for mdx production of color table and line plots (advanced feature)

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
-------------------------------------------------------
* cython3 - must have an executable named cython3 (use a symbolic link)
* cuda - for GPUtopozero and GPUgeo2rdr


1.1 Installing pyre and config
------------------------------

Detailed instructions for installing pyre and config are found at:

pyre.orthologue.com/install


1.2 Installing software dependencies with standard package managers
-------------------------------------------------------------------

The easiest way to install most of these is with package managers such as
'apt-get' on Linux systems or 'macports' on MacOsX or anaconda.  To use these,
however, may require that you have superuser permission on your computer. The
following URL gives additional information on installing prerequisites for
ISCE:

https://winsar.unavco.org/portal/wiki/Manual%20installation%20using%20repository%20managers/

If it is not possible for you to install the software yourself and you
can't convince the System Administrator on your computer to install the
dependencies, then we provide virtual machine images (VMs) with the
dependencies pre-installed (see Section 1.2).

For the truly adventurous who want to install dependencies by hand we provide
some hints in Section 1.5.

When you have installed the dependencies you can skip the other sections about
installing the dependencies and read Section 1.6 about the 'python3' convention
and then Section 2 on building ISCE and configuring your environment.


1.3 Installing Virtual Machine Images with Dependencies Pre-Installed
---------------------------------------------------------------------

If you don't have superuser privileges on your machine and your system is not
up to date with the software dependencies required to use ISCE, then you can
download Virtual Machine Images (VMs) at the following URL:

Full link: http://earthdef.caltech.edu/boards/4/topics/305
Simple link: http://tinyurl.com/iscevm

Instructions on how to install the Virtual Machines are given there.


1.4 Installing dependencies with provided setup script
------------------------------------------------------

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


1.5 Hints for installing dependencies by hand.
----------------------------------------------

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
Get the Python source code from http://www.python.org/ftp/python/3.3.5/Python-3.3.5.tgz

Untar the file Python-3.3.5.tgz using

> tar -zxvf Python-3.3.5.tgz
> cd Python-3.3.5

Then run

> ./configure --prefix=<directory>

where <directory> is the full path to an installation location where you
have write access.  Then run,

> make
> make install

Building hdf5 [Only necessary for COSMO-SkyMed support]
-------------
Get the source code from:
http://www.hdfgroup.org/ftp/HDF5/current/src/hdf5-1.8.6.tar.gz

Building h5py [Only necessary for COSMO-SkyMed support]
Get the h5py source code from:
http://code.google.com/p/h5py/downloads/detail?name=h5py-1.3.1.tar.gz


Building gdal-bindings [Only necessary for Radarsat2, Tandem-X and Sentinel 1A]
----------------------
On most linux distributions, gdal can installed along with its python bindings
using standard repository management tools.

If you don't have gdal, you can find instructions on building GDAL here
http://trac.osgeo.org/gdal/wiki/BuildHints

Remember to use configure with --with-python option to automatically build
python bindings.

Else, if you already have gdal installed on your system but without python
buindings, use easy_install corresponding to your python 3 version. You may
need to setup LD_LIBRARY_PATH correctly to include path to libgdal.so

> easy_install GDAL


Building SpiceyPy  [Only necessary for Radarsat1]
-----------------
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


1.6 Note On 'python3' Executable Convention
-------------------------------------------

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


2.0 Building ISCE
=================


2.1 Working with config's mm
----------------------------

The config package builder uses the command "mm" to build and install the pyre
and isce codes. To add non-standard paths to mm, such as the location of pyre,
run the command 'mm.paths' while in the directory to be added.  For instance,
pyre-1.0 is a dependency whose path must be know to mm before building isce.
Go to the pyre-1.0 top directory and then issue the command, "mm.paths" at the
command line.  That is all that is necessary to inform mm of the path.


2.2 Installing ISCE with mm
---------------------------

To install ISCE, simply enter the top level of the ISCE directory and issue the
command "mm".  The directory products will contain the installed libraries,
headers, and Python applications and components.


2.3 Set up your environment
---------------------------

Add the isce "products/packages" directory to your PYTHONPATH.


3.0 Running ISCE
================


3.1 Running ISCE from the command line
--------------------------------------

Copy the example xml files located in the example directory in the ISCE source
tree to a working directory and modify them to point to your own data.  Run
them using the command:

> $ISCE_HOME/applications/insarApp.py isceInputFile.xml

or (with $ISCE_HOME/applications on your PATH) simply,

> insarApp.py isceInputFile.xml

The name of the input file on the command line is arbitrary.  ISCE also looks
for appropriately named input files in the local directory

You can also ask ISCE for help from the command line:

> insarApp.py --help

This will tell you the basic command and the options for the input file.
Example input files are also given in the 'examples/input_files' directory.

As explained in the Component Configurability section below, it is also possible
to run insarApp.py without giving an input file on the command line.  ISCE will
automatically find configuration files for applications and components if they
are named appropriately.

---------------------------------------------------------------------------------
3.2 Running ISCE in the Python interpreter
---------------------------------------------------------------------------------

It is also possible to run ISCE from within the Python interpreter.  If you have
an input file named insarInputs.xml you can do the following:

```
%> python3
>>> import isce
>>> from insarApp import Insar
>>> a = Insar(name="insarApp", cmdline="insarInputs.xml")
>>> a.configure()
>>> a.run()
```

(As explained in the Component Configurability section below, if the file
insarInputs.xml were named insarApp.xml or insar.xml, then the 'cmdline' input
on the line creating 'a' would not be necessary.  The file 'insarApp.xml' would
be loaded automatically because when 'a' is created above it is given the name
'insarApp'.  A file named 'insar.xml' would also be loaded automatically if it
exists because the code defining insarApp.py gives all instances of it the
'family' name 'insar'.  See the Component Configurability section below for
details.)


3.3 Running ISCE with steps
---------------------------

An other way to run ISCE is the following:

> insarApp.py insar.xml --steps

This will run insarApp.py from beginning to end as is done without the
--steps option, but with the added feature that the workflow state is
stored in files after each step in the processing using Python's pickle
module. This method of running insarApp.py is only a little slower
and it uses extra disc space to store the pickle files, but it
provides some advantage for debugging and for stopping and starting a
workflow at any predetermined point in the flow.

The full options for running insarApp.py with steps is the following:

```
> insarApp.py insar.xml [--steps] [--start=<s>] [--end=<s>] [--dostep=<s>]
```

where ```<s>``` is the name of a step.  To see the full ordered list of steps
the user can issue the following command:

> insarApp.py insar.xml --steps --help

The --steps option was explained above.
The --start and --end option can be used together to process a range of steps.
The --dostep option is used to process a single step.

For the --start and --dostep options to work, of course, requires that the
steps preceding the starting step have been run previously because the
state of the work flow at the beginning of the first step to be run must
be stored from a previous run.

An example for using steps might be to execute the end-to-end workflow
with --steps to store the state of the workflow after every step as in,

> insarApp.py insar.xml --steps

Then use --steps to rerun some of the steps (perhaps you made a code
modification for one of the steps and want to test it without starting
from the beginning) as in

> insarApp.py insar.xml --start=<step-name1> --end=<step-name2>

or to rerun a single step as in

> insarApp.py insar.xml --dostep=<step-name>

Running insarApp.py with --steps also enables one to enter the Python
interpreter after a run and load the state of the workflow at any stage
and introspect the objects in the flow and play with them as follows,
for example:

```
> python3
>>> import isce
>>> f = open("PICKLE/formslc")
>>> import pickle
>>> a = pickle.load(f)
>>> o = f.getMasterOrbit()
>>> t, x, p, off = o._unpackOrbit()
>>> print t
>>> print x
>>>
```

Someone with familiarity of the inner workings of ISCE can exploit
this mode of interacting with the pickle object to discover much about
the workflow states and also to edit the state to see its effect
on a subsequent run with --dostep or --start.


3.4 NOTE on DEM
---------------

- If a dem component is provided but the dem is the EGM96 geo reference
  (which is the case for SRTM DEMs) it will be converted into  WGS84.
  A new file with suffix wgs84 is created. If it is already in WGS84
  nothing happens.

- If no dem component is specified in inputs a EGM96 will be downloaded
  and then it will be converted into WGS84. There will be then two files,
  an EGM96 with no suffix, and the WGS84 with the wgs84 suffix.

- To enable automatic downloading of dems, you need to have a user name
and password from urs.earthdata.nasa.gov and you need to include LPDAAC
applications to your account.
* a. If you don't already have an earthdata username and password,
you can set them at https://urs.earthdata.nasa.gov/
* b. If you already have an earthdata account, please ensure that
you add LPDAAC applications to your account:
* Login to earthdata here: https://urs.earthdata.nasa.gov/home
* Click on my applications on the profile
* Click on “Add More Applications”
* Search for “LP DAAC”
* Select “LP DAAC Data Pool” and “LP DAAC OpenDAP” and approve.

- create a file in your HOME directory named **.netrc** with the following 3
lines:

```
machine urs.earthdata.nasa.gov
   login your_earthdata_login_name
   password your_earthdata_password
```

3. set permissions to prevent others from viewing your credentials:

> chmod go-rwx .netrc


- A note on making stitched dems and water masks globally available. Stitched
dems and water body masks will be stored in and used from a directory indicated
by the environment variable, $DEMDB, if you have defined this environment
variable with a value pointing to a path where you want to store stitched dems
and waterbody masks. The stitched dem or water mask will be globally available
automatically (so long as the $DEMDB environment variable is valid) without
needing to specify any information about the dem in your input files for ISCE
processing applications. If you use dem.py or watermask.py, the stitched
products are left in the directory where you run these apps.  If you want them
to be globally available, then either run demdb.py or wbd.py (which will place
the dems and water body masks in the $DEMDB directory if defined).  If you
move dems and waterbody masks to the $DEMDB directory from other directories
you will need to modify their meta data files (.xml) to indicate the new
location.


4.0 Input Files
===============

Input files are structured 'xml' documents.  This section will briefly
introduce their structure using a special case appropriate for processing ALOS
data.  Examples for the other sensor types can be found in the directory
'examples/input_files'.

The basic (ALOS) input file looks like this (indentation is optional):

insarApp.xml:
-------------
```
<insarApp>
 <component name="insarApp">
     <property name="sensor name">ALOS</property>
     <component name="Master">
         <property name="IMAGEFILE">
             /a/b/c/20070215/IMG-HH-ALPSRP056480670-H1.0__A
         </property>
         <property name="LEADERFILE">
             /a/b/c/20070215/LED-ALPSRP056480670-H1.0__A
         </property>
         <property name="OUTPUT">20070215.raw </property>
     </component>
     <component name="Slave">
         <property name="IMAGEFILE">
             /a/b/c/20061231/IMG-HH-ALPSRP049770670-H1.0__A
         </property>
         <property name="LEADERFILE">
             /a/b/c/20061231/LED-ALPSRP049770670-H1.0__A
         </property>
         <property name="OUTPUT">20061231.raw </property>
     </component>
 </component>
</insarApp>
```

The data are enclosed between an opening tag and a closing tag.  The <insarApp>
tag is closed by the </insarApp> tag for example.  This outer tag is necessary
but its name has no significance.  You can give it any name you like.  The
other tags, however, need to have the names shown above.  There are 'property',
and 'component' tags shown in this example.

The component tags have names that match a Component name in the ISCE code.
The component tag named 'insarApp' refers to the configuration information for
the Application (which is a Component) named "insarApp".  Components contain
properties and other components that are configurable.  The property tags
give the values of a single variable in the ISCE code.  One of the properties
defined in insarApp.py is the "sensor name" property.  In the above example
it is given the value ALOS.  In order to run insarApp.py two images need to
be specified.  These are defined as components named 'Master' and 'Slave'.
These components have properties named 'IMAGEFILE', 'LEADERFILE', and 'OUTPUT'
with the values given in the above example.

NOTE: the capitalization of the property and component names are not of any
importance.  You could enter 'imagefile' instead of 'IMAGEFILE', for example,
and it would work correctly.  Also extra spaces in names that include spaces,
such as "sensor name" do not matter.

There is a lot of flexibility provided by ISCE when constructing these input
files through the use of "catalog" tags and "constant" tags.

A "catalog" tag can be used to indicate that the contents that would normally
be found between an opening ad closing "component" tag are defined in another
xml file.  For example, the insarApp.xml file shown above could have been split
between three files as follows:

insarApp.xml
------------
```
<insarApp>
    <component name="insar">
        <property  name="Sensor name">ALOS</property>
        <component name="master">
            <catalog>20070215.xml</catalog>
        </component>
        <component name="slave">
            <catalog>20061231.xml</catalog>
        </component>
    </component>
</insarApp>
```

20070215.xml
------------
```
<component name="Master">
    <property name="IMAGEFILE">
        /a/b/c/20070215/IMG-HH-ALPSRP056480670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /a/b/c/20070215/LED-ALPSRP056480670-H1.0__A
    </property>
    <property name="OUTPUT">20070215.raw </property>
</component>
```

20061231.xml
------------
```
<component name="Slave">
    <property name="IMAGEFILE">
        /a/b/c/20061231/IMG-HH-ALPSRP049770670-H1.0__A
    </property>
    <property name="LEADERFILE">
        /a/b/c/20061231/LED-ALPSRP049770670-H1.0__A
    </property>
    <property name="OUTPUT">20061231.raw</property>
</component>
```

A "constant" tag can be used to define a constant for convenience inside
an xml file.  For example, the dates '20070215' and '20061231' are used
multiple times in the above files.  Also, the base path '/a/b/c/' is used
multiple times.  A constant defined in a constant tag is used in constructing
values by sandwiching it between two '$' symbols.  For example, if a constant
named "date1" is defined then to use it we would enter '$date1'.  The following
example insarApp.xml file should make this clear:

insarApp.xml
------------
```
<insarApp>
<constant name="dir">/a/b/c </constant>
<constant name="date1">20070215</constant>
<constant name="date2">20061231</constant>
<constant name="dir1">$dir$/$date1$</constant>
<constant name="dir2">$dir$/$date2$</constant>
<component name="insarApp">
    <property name="sensor name">ALOS</property>
    <component name="Master">
        <property name="IMAGEFILE">
            $dir1$/IMG-HH-ALPSRP056480670-H1.0__A
        </property>
        <property name="LEADERFILE">
            $dir1$/LED-ALPSRP056480670-H1.0__A
        </property>
        <property name="OUTPUT">$date1$.raw </property>
    </component>
    <component name="Slave">
        <property name="IMAGEFILE">
            $dir2$/IMG-HH-ALPSRP049770670-H1.0__A
        </property>
        <property name="LEADERFILE">
            $dir2$/LED-ALPSRP049770670-H1.0__A
        </property>
        <property name="OUTPUT">$date2$.raw </property>
    </component>
</component>
</insarApp>
```

Note: as of the time of this release constants do not work with catalog files.
This will be fixed in a future release.


5.0 Component Configurability
=============================

In the examples for running insarApp.py (Section 3.1 and 3.3 above) the input
data were entered by giving the name of an 'xml' file on the command line.  The
ISCE framework parses that 'xml' file to assign values to the configurable
variables in the isce Application insarApp.py.  The Application executes
several steps in its workflow.  Each of those steps are handled by a Component
that is also configurable from input data.  Each component may be configured
independently from user input using appropriately named and placed xml files.
This section will explain how to name these xml files and where to place them.

----------------------------------------------------------------------------
5.1 Component Names: Family and Instance
----------------------------------------------------------------------------

Each configurable component has two "names" associated with it.  These names
are used in locating possible configuration xml files for those components. The
first name associated with a configurable component is its "family" name.  For
insarApp.py, the family name is "insar".  Inside the insarApp.py file an
Application is created from a base class named Insar.  That base class defines
the family name "insar" that is given to every instance created from it.  The
particular instance that is created in the file insarApp.py is given the
'instance name' 'insarApp'.  If you look in the file near the bottom you will
see the line,

> insar = Insar(name="insarApp")

This line creates an instance of the class Insar (that is given the family name
'insar' elsewhere in the file) and gives it the instance name "insarApp".

Other applications could be created that could make several different instances
of the Insar.  Each instance would have the family name "insar" and would be
given a unique instance name.  This is possible for every component.  In the
above example xml files instances name "Master" and "Slave" of a family named
"alos" are created.

----------------------------------------------------------------------------
5.2 Component Configuration Files: Locations, Names, Priorities
----------------------------------------------------------------------------

The ISCE framework looks for xml configuration files when configuring every
Component in its flow in 3 different places with different priorities.  The
configuration sequence loads configuration parameters found in these xml files
in the sequence lowest to highest priority overwriting any parameters defined
as it moves up the priority sequence.  This layered approach allows a couple
of advantages.  It allows the user to define common parameters for all instances
in one file while defining specific instance parameters in files named for those
specific instances.  It also allows global preferences to be set in a special
directory that will apply unless the user overrides them with a higher priority
xml file.

The priority sequence has two layers.  The first layer is location of the xml
file and the second is the name of the file.  Within each of the 3 location
priorities indicated below, the filename priority goes from 'family name' to
'instance name'.  That is, within a given location priority level, a file
named after the 'family name' is loaded first and then a file with the
'instance name' is loaded next and overwrites any property values read from the
'family name' file.

The priority sequence for location is as follows:

(1)  The highest priority location is on the command line.  On the command line
the filename can be anything you choose.  Configuration parameters can also be
entered directly on the command line as in the following example:

> insarApp.py insar.master.output=master_c.raw

This example indicates that the variable named 'output' of the Component
named 'master' belonging to the Component (or Application) named 'insar'
will be given the name "master_c.raw".

The priority sequence on the command line goes from lowest priority on the left
to highest priority on the right.  So, if we use the command line,

> insarApp.py myInputFile.xml insar.master.output=master_c.raw

where the myInputFile.xml file also gives a value for the insar master output
file as master_d.raw, then the one defined on the right will win, i.e.,
master_c.raw.

(2) The next priority location is the local directory in which insarApp.py is
executed.  Any xml file placed in this directory named according to either the
family name or the instance name for any configurable component in ISCE will be
read while configuring the component.

(3) If you define an environment variable named $ISCEDB, you can place xml files
with family names or instance names that will be read when configuring
Configurable Components.  These files placed in the $ISCEDB directory have the
lowest priority when configuring properties of the Components.  The files placed
in the ISCEDB directory can be used to define global settings that will apply
unless the xml files in the local directory or the command line override those
preferences.

----------------------------------------------------------------------------
5.2 Component Configuration Structure
----------------------------------------------------------------------------

However, the component tag has to have the family name of the Component/
Application.  In the above examples you see
that the outermost component tag has the name "insar", which is the family name
of the class Insar of which insarApp is an instance.


----------------------------------------------------------------------------
5.3 Component Configuration Help
----------------------------------------------------------------------------

At this time there is limited information about component configurability
through the command

> insarApp.py --help

Future deliveries will improve this situation.  In the meantime we describe
here how to discover from the code which Components and parameters are
configurable.  One note of caution is that it is possible for a parameter
to appear to be configurable from user input when the particular flow will
not allow this degree of freedom.  Experience and evolving documentation will
be of use in determining these cases.

How to find out whether a component is configurable, what its configurable
parameters are, what "name" to use in the xml file, and what name to give to
the xml file.

Let's take as an example, Nstage.py, which is in components/mroipac/ampcor.

Open it in an editor and search for the string "class Nstage".  It is on
line 243.  You will see that it inherits from Component.  This is the minimum
requirement for it to be a configurable component.

Now look above that line and you will see several variable names being set
equal to a call to Component.Parameter.  These declarations define these
variables as configurable parameters.  They are entered in the "parameter_list"
starting on line 248.  That is the method by which these Parameters are made
configurable parameters of the Component Nstage.

Each of the parameters defines the "public_name", which is the "name" that you
would enter in the xml file.  For instance if you want to set the gross offset
in range, which is defined starting on line 130 in the variable
ACROSS_GROSS_OFFSET, then you would use an xml tag like the following (assuming
you have determined that the gross offset in range is about 150 pixels):

> <property name="ACROSS_GROSS_OFFSET">150</property>


Now, to determine what to call the xml file and what "name" to use in the
component tag.  A configurable component has a "family" name and an instance
"name".  It is registered as having these names by calling the
Component.__init__ constructor, which is done on line 672.  On that line you
will see that the call to __init__ passes  'family=self.__class__.family' and
'name=name' to the Component constructor (super class of Nstage).  The family
name is given as "nstage" on line 245.  The instance name is passed as the
value of the 'name=name' and was passed to it from whatever program created it.
Nstage is created in  components/isceobj/InsarProc/runOffsetprf_nstage.py where
it is given the name 'insarapp_slcs_nstage'  on line 107 and also in
components/isceobj/InsarProc/runRgoffset_nstage.py where it is given the name
'insarapp_intsim_nstage' on line 58.  If you are setting a parameter that
should be the same for both of these uses of Nstage, then you can use the
family name 'nstage' for the name of the xml file as 'nstage.xml'.  It is more
likely that you will want to use the instance names, 'insarapp_slcs_nstage.xml'
and 'insarapp_intsim_nstage.xml'.  Use the family name 'nstage' for the
component tag 'name'.

Example for SLC matching use of Nstage:

Filename: insarapp_slcs_nstage.xml:

```
<dummy>
<component name="nstage">
    <property name="ACROSS_GROSS_OFFSET">150</property>
</component>
</dummy>
```


END OF FILE
===========
