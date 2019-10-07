# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os, re
# access to the framework
import pyre
# superclass
from .Library import Library


# the petsc package manager
class PETSc(Library, family='pyre.externals.petsc'):
    """
    The package manager for PETSc packages
    """

    # constants
    category = 'petsc'


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
        rgx = r"lib{cls.category}(?P<version>.*)-dev$".format(cls=cls)
        # the recognizer of python dev packages
        scanner = re.compile(rgx)

        # go through the names of all installed packages
        for key in installed.keys():
            # looking for ones that match my pattern
            match = scanner.match(key)
            # once we have a match
            if match:
                # extract the variant
                version = ''.join(match.group('version').split('.'))
                # fold it into the installation name
                name = cls.category + version
                # place the package name into a tuple
                package = match.group(),
                # hand them to the caller
                yield name, package

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Identify the default implementation of PETSc on dpkg machines
        """
        # ask {dpkg} for my options
        alternatives = sorted(packager.alternatives(group=cls), reverse=True)
        # the supported versions
        versions = Default,
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
        Identify the default implementation of PETSc on macports machines
        """
        # there is only one variation of this
        yield Default(name=cls.category)
        # and nothing else
        return


# superclass
from .LibraryInstallation import LibraryInstallation


# the implementation
class Default(LibraryInstallation, family='pyre.externals.petsc.default', implements=PETSc):
    """
    A generic PETSc installation
    """

    # constants
    category = PETSc.category
    flavor = category

    # public state
    defines = pyre.properties.strings(default="WITH_PETSC")
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

        # the package leaves includes in
        self.incdir = [ '/usr/include/petsc' ]
        # and libraries in
        self.libdir = [ '/usr/lib/x86_64-linux-gnu' ]
        # set my library
        self.libraries = 'petsc'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


    def macports(self, packager, **kwds):
        """
        Attempt to repair my configuration
        """
        # the name of the macports package
        package = self.category
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
        header = 'petsc.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = self.flavor
        # convert it into the actual file name
        libpetsc = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libpetsc, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.incdir+self.libdir)

        # all done
        return


# end of file
