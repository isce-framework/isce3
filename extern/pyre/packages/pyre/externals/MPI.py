# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# superclass
from .Tool import Tool
from .Library import Library


# the mpi package manager
class MPI(Tool, Library, family='pyre.externals.mpi'):
    """
    The package manager for MPI packages
    """

    # constants
    category = 'mpi'

    # user configurable state
    launcher = pyre.properties.str(default='mpirun')
    launcher.doc = 'the name of the launcher of MPI jobs'


    # support for specific package managers
    @classmethod
    def dpkgAlternatives(cls, dpkg):
        """
        Identify the default implementation of MPI on dpkg machines
        """
        # get the index of installed packages
        installed = dpkg.installed()

        # the OpenMPI development packages
        openmpi = "openmpi-bin", "libopenmpi-dev"
        # find the missing ones
        missing = [ pkg for pkg in openmpi if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield OpenMPI.flavor, openmpi

        # the MPICH development packages
        mpich = "mpich", "libmpich-dev"
        # find the missing ones
        missing = [ pkg for pkg in mpich if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield MPICH.flavor, mpich

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Provide alternative compatible implementations of python on dpkg machines, starting
        with the package the user has selected as the default
        """
        # ask {dpkg} for my options
        # the supported versions
        flavors = OpenMPI, MPICH
        # go through the versions
        for flavor in flavors:
           # scan through the alternatives
            for alternative in sorted(packager.alternatives(group=cls), reverse=True):
                # if it is match
                if alternative.startswith(flavor.flavor):
                    # build an instance and return it
                    yield flavor(name=alternative)

        # out of ideas
        return


    @classmethod
    def macportsPackages(cls, packager):
        """
        Provide alternative compatible implementations of MPI on macports machines, starting with
        the package the user has selected as the default
        """
        # build a locator
        locator = pyre.tracking.simple('while looking for {.category!r} choices'.format(cls))
        # known installations
        flavors = OpenMPI, MPICH
        # on macports, mpi is a package group
        for alternative in packager.alternatives(group=cls.category):
            # go through the known installations
            for flavor in flavors:
                # if the package name starts with the installation flavor
                if alternative.startswith(flavor.flavor):
                    # instantiate the package and return it
                    yield flavor(name=alternative, locator=locator)
        # out of ideas
        return


# superclass
from .ToolInstallation import ToolInstallation
from .LibraryInstallation import LibraryInstallation


# the base class
class Default(ToolInstallation, LibraryInstallation, implements=MPI):
    """
    The package manager for unknown MPI installations
    """

    # constants
    flavor = "unknown"
    category = MPI.category

    # public state
    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'

    launcher = pyre.properties.str()
    launcher.doc = 'the name of the launcher of MPI jobs'


    # configuration
    def macports(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask macports for help; start by finding out which package supports me
        package = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=package)
        # and the package contents
        contents = tuple(packager.contents(package=package))

        # {mpi} is a selection group
        group = self.category
        # the package deposits its selection alternative here
        selection = str(packager.prefix() / 'etc' / 'select' / group / '(?P<alternate>.*)')
        # so find it
        match = next(packager.find(target=selection, pile=contents))
        # extract the name of the alternative
        alternative = match.group('alternate')
        # ask for the normalization data
        normalization = packager.getNormalization(group=group, alternative=alternative)
        # build the normalization map
        nmap = { base: target for base,target in zip(*normalization) }
        # find the binary that supports {mpirun} and use it to set my launcher
        self.launcher = nmap[pyre.primitives.path('bin/mpirun')].name
        # extract my {bindir}
        bindir = packager.findfirst(target=self.launcher, contents=contents)
        # and save it
        self.bindir = [ bindir ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = 'mpi.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        libmpi = self.pyre_host.dynamicLibrary('mpi')
        # find it
        libdir = packager.findfirst(target=libmpi, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = 'mpi_cxx', 'mpi'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.bindir+self.incdir+self.libdir)

        # all done
        return


# the openmpi package manager
class OpenMPI(Default, family='pyre.externals.mpi.openmpi'):
    """
    The package manager for OpenMPI packages
    """

    # constants
    flavor = "openmpi"
    category = MPI.category

    # public state
    defines = pyre.properties.strings(default="WITH_OPENMPI")
    defines.doc = "the compile time markers that indicate my presence"


    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask dpkg for help; start by finding out which package supports me
        bin, dev = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=dev)

        # the name of the launcher
        self.launcher = 'mpirun.{.flavor}'.format(self)
        # our search target for the bindir is in a bin directory to avoid spurious matches
        launcher = "bin/{.launcher}".format(self)
        # find it in order to identify my {bindir}
        bindir = packager.findfirst(target=launcher, contents=packager.contents(package=bin))
        # and save it
        self.bindir = [ bindir / 'bin' ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = r'mpi\.h'
        # find it
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = '{0.category}'.format(self)
        # convert it into the actual file name
        libpython = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libpython, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.bindir+self.incdir+self.libdir)

        # all done
        return


# the mpich package manager
class MPICH(Default, family='pyre.externals.mpi.mpich'):
    """
    The package manager for MPICH packages
    """

    # constants
    flavor = "mpich"
    category = MPI.category

    # public state
    defines = pyre.properties.strings(default="WITH_MPICH")
    defines.doc = "the compile time markers that indicate my presence"


    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask dpkg for help; start by finding out which package supports me
        bin, dev = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=dev)

        # the name of the launcher
        self.launcher = 'mpirun.{.flavor}'.format(self)
        # our search target for the bindir is in a bin directory to avoid spurious matches
        launcher = "bin/{.launcher}".format(self)
        # find it in order to identify my {bindir}
        bindir = packager.findfirst(target=launcher, contents=packager.contents(package=bin))
        # and save it
        self.bindir = [ bindir / 'bin' ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = r'mpi\.h'
        # find it
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor # on dpkg, mpich doesn't come with a libmpi.so
        # convert it into the actual file name
        libpython = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libpython, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.bindir+self.incdir+self.libdir)

        # all done
        return


# end of file
