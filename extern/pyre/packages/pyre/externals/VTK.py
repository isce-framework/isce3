# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re
# access to the framework
import pyre
# superclass
from .Library import Library


# the vtk package manager
class VTK(Library, family='pyre.externals.vtk'):
    """
    The package manager for VTK packages
    """

    # constants
    category = 'vtk'


    # support for specific package managers
    @classmethod
    def dpkgAlternatives(cls, dpkg):
        """
        Go through the installed packages and identify those that are relevant for providing
        support for my installations
        """
        # get the index of installed packages
        installed = dpkg.installed()
        # the package regex
        rgx = r"^libvtk(?P<version>[0-9])-dev$"
        # the recognizer of python dev packages
        scanner = re.compile(rgx)

        # go through the names of all installed packages
        for key in installed.keys():
            # looking for ones that match my pattern
            match = scanner.match(key)
            # once we have a match
            if match:
                # extract the version
                version = match.group('version')
                # fold it into the installation name
                name = cls.category + version
                # place the package name into a tuple
                packages = match.group(),
                # hand them to the caller
                yield name, packages

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Identify the default implementation of HDF5 on dpkg machines
        """
        # ask {dpkg} for my options
        alternatives = sorted(packager.alternatives(group=cls), reverse=True)
        # the supported versions
        versions = VTK6, VTK5
        # go through the versions
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
        Identify the default implementation of VTK on macports machines
        """
        # version 6.x installations
        yield VTK6(name=cls.category)
        # version 5.x installations
        yield VTK5(name=cls.category+'5')
        # and nothing else
        return


# superclass
from .LibraryInstallation import LibraryInstallation


# the implementation
class VTK5(LibraryInstallation, family='pyre.externals.vtk.vtk5', implements=VTK):
    """
    Support for VTK 5.x installations
    """

    # constants
    category = VTK.category
    flavor = category + '5'

    # public state
    defines = pyre.properties.strings(default="WITH_VTK6")
    defines.doc = "the compile time markers that indicate my presence"

    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'


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
        header = 'vtkVersion.h'
        # find the header
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = 'CommonCore'
        # convert it into a library
        libvtk = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libvtk, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


    def macports(self, packager, **kwds):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = 'vtk5'
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
        # grab the package contents
        contents = tuple(packager.contents(package=package))

        # in order to identify my {incdir}, search for the top-level header file
        header = 'vtkVersion.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = 'vtkCommonCore'
        # convert it into a library
        libvtk = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libvtk, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


# the implementation
class VTK6(LibraryInstallation, family='pyre.externals.vtk.vtk6', implements=VTK):
    """
    Support for VTK 6.x installations
    """

    # constants
    category = VTK.category
    flavor = category + '6'

    # public state
    defines = pyre.properties.strings(default="WITH_VTK6")
    defines.doc = "the compile time markers that indicate my presence"

    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'


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
        header = 'vtkVersion.h'
        # find the header
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.libgen('CommonCore')
        # convert it into a library
        libvtk = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libvtk, contents=packager.contents(package=dev))
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


    def macports(self, packager, **kwds):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = 'vtk'
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
        self.version, _ = packager.info(package=package)
        # and the package contents
        contents = tuple(packager.contents(package=package))

        # in order to identify my {incdir}, search for the top-level header file
        header = 'vtkVersion.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.libgen('CommonCore')
        # convert it into a library
        libvtk = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libvtk, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


    # interface
    def libgen(self, stem):
        """
        Construct the name of a library given a capability {stem}
        """
        # build and return
        return 'vtk{}-{.sigver}'.format(stem, self)


# end of file
