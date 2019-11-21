# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# superclass
from .Library import Library


# the blas package manager
class BLAS(Library, family='pyre.externals.blas'):
    """
    The package manager for BLAS packages
    """

    # constants
    category = 'blas'


    # support for specific package managers
    @classmethod
    def dpkgAlternatives(cls, dpkg):
        """
        Go through the installed packages and identify those that are relevant for providing
        support for my installations
        """
        # get the index of installed packages
        installed = dpkg.installed()

        # the ATLAS development packages
        atlas = 'libatlas-base-dev', 'libatlas-dev'
        # find the missing ones
        missing = [ pkg for pkg in atlas if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield Atlas.flavor, atlas

        # the OpenBLAS development packages
        openblas = 'libopenblas-dev', 'libopebblas-base'
        # find the missing ones
        missing = [ pkg for pkg in atlas if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield OpenBLAS.flavor, openblas

        # the GSL development packages
        gsl = 'libgsl0-dev',
        # find the missing ones
        missing = [ pkg for pkg in atlas if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield GSLCBLAS.flavor, gsl

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Identify the default implementation of BLAS on dpkg machines
        """
        # ask {dpkg} for my options
        alternatives = sorted(packager.alternatives(group=cls), reverse=True)
        # the order of preference of these implementations
        versions = Atlas, OpenBLAS, GSLCBLAS
        # go through each one
        for version in versions:
           # scan through the alternatives
            for name in alternatives:
                # if it is match
                if name.startswith(version.flavor):
                    # build an instance and return it
                    yield version(name=name)

        # out of ideas
        return


    @classmethod
    def macportsPackages(cls, packager):
        """
        Identify the default implementation of BLAS on macports machines
        """
        # on macports, the following packages provide support for BLAS, ranked by their
        # performance: atlas, openblas, gsl
        versions = Atlas, OpenBLAS, GSLCBLAS
        # get the index of installed packages
        installed = packager.getInstalledPackages()
        # go through each one
        for version in versions:
            # get the flavor
            flavor = version.flavor
            # look for an installation
            if flavor in installed:
                # build an instance and return it
                yield version(name=flavor)

        # out of ideas
        return


# superclass
from .LibraryInstallation import LibraryInstallation


# the base class
class Default(LibraryInstallation, family='pyre.externals.blas.default', implements=BLAS):
    """
    A generic BLAS installation
    """

    # constants
    flavor = 'unknown'
    category = BLAS.category

    # public state
    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'


# atlas
class Atlas(Default, family='pyre.externals.blas.atlas'):
    """
    Atlas BLAS support
    """

    # constants
    flavor = 'atlas'

    # public state
    defines = pyre.properties.strings(default="WITH_ATLAS")
    defines.doc = "the compile time markers that indicate my presence"


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # get the names of the packages that support me
        lib, headers = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=lib)

        # in order to identify my {incdir}, search for the top-level header file
        header = 'atlas/atlas_buildinfo.h'
        # find the header
        incdir = packager.findfirst(target=header, contents=packager.contents(package=headers))
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor
        # convert it into the actual file name
        libatlas = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libatlas, contents=packager.contents(package=lib))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)
        # all done
        return


    def macports(self, packager):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = 'atlas'
        # attempt to
        try:
            # get the version info
            self.version, _ = packager.info(package=package)
        # if this fails
        except KeyError:
            # this package is not installed
            msg = 'the package {!r} is not installed'.format(package)
            # complain
            raise self.ConfigurationError(configurable=self, errors=[msg])
        # otherwise, grab the package contents
        contents = tuple(packager.contents(package=package))

        # in order to identify my {incdir}, search for the top-level header file
        header = 'atlas/atlas_buildinfo.h'
        # look for it
        incdir = packager.findfirst(target=header, contents=contents)
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        libatlas = self.pyre_host.staticLibrary('atlas')
        # look for
        libdir = packager.findfirst(target=libatlas, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library list
        self.libraries = 'cblas', 'atlas'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


# OpenBLAS
class OpenBLAS(Default, family='pyre.externals.blas.openblas'):
    """
    OpenBLAS support
    """

    # constants
    flavor = 'openblas'

    # public state
    defines = pyre.properties.strings(default="WITH_OPENBLAS")
    defines.doc = "the compile time markers that indicate my presence"


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # get the names of the packages that support me
        dev, *_ = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=dev)

        # in order to identify my {incdir}, search for the top-level header file
        header = 'openblas/openblas_config.h'
        # find the header
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor
        # convert it into the actual file name
        libatlas = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libatlas, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)
        # all done
        return


    def macports(self, packager):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = 'OpenBLAS'
        # attempt to
        try:
            # get the version info
            self.version, _ = packager.info(package=package)
        # if this fails
        except KeyError:
            # this package is not installed
            msg = 'the package {!r} is not installed'.format(package)
            # complain
            raise self.ConfigurationError(configurable=self, errors=[msg])
        # otherwise, grab the package contents
        contents = tuple(packager.contents(package=package))

        # in order to identify my {incdir}, search for the top-level header file
        header = 'cblas_openblas.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        libopenblas = self.pyre_host.dynamicLibrary('openblas')
        # find it
        libdir = packager.findfirst(target=libopenblas, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = 'openblas'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


# gslcblas
class GSLCBLAS(Default, family='pyre.externals.blas.gslcblas'):
    """
    GSL BLAS support
    """

    # constants
    flavor = 'gslcblas'

    # public state
    defines = pyre.properties.strings(default="WITH_GSLCBLAS")
    defines.doc = "the compile time markers that indicate my presence"


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # get the names of the packages that support me
        dev, *_ = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=dev)

        # in order to identify my {incdir}, search for the top-level header file
        header = 'gsl/gsl_cblas.h'
        # find the header
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor
        # convert it into the actual file name
        libgsl = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libgsl, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


    def macports(self, packager):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = 'gsl'
        # attempt to
        try:
            # get the version info
            self.version, _ = packager.info(package=package)
        # if this fails
        except KeyError:
            # this package is not installed
            msg = 'the package {!r} is not installed'.format(package)
            # complain
            raise self.ConfigurationError(configurable=self, errors=[msg])
        # otherwise, grab the package contents
        contents = tuple(packager.contents(package=package))

        # in order to identify my {incdir}, search for the top-level header file
        header = 'gsl/gsl_cblas.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor
        # convert it into the actual file name
        libgsl = self.pyre_host.dynamicLibrary('gslcblas')
        # find it
        libdir = packager.findfirst(target=libgsl, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = 'gslcblas'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


# end of file
