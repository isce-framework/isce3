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
from .Tool import Tool
from .Library import Library


# the postgres package manager
class Postgres(Tool, Library, family='pyre.externals.postgres'):
    """
    The package manager for postgres client development
    """

    # constants
    category = 'postgresql'

    # user configurable state
    psql = pyre.properties.str()
    psql.doc = 'the full path to the postgres client'


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
        rgx = r"^postgresql-client-(?P<version>(?P<major>[0-9]+)(\.(?P<minor>[0-9]+))?)$"
        # the recognizer of python dev packages
        scanner = re.compile(rgx)

        # go through the names of all installed packages
        for key in installed.keys():
            # looking for ones that match my pattern
            match = scanner.match(key)
            # once we have a match
            if match:
                # and the sequence of packages
                packages = match.group(), 'libpq-dev'
                # find the missing ones
                missing = [ pkg for pkg in packages if pkg not in installed ]
                # if there are no missing ones
                if not missing:
                    # extract the version and collapse it
                    version = ''.join(match.group('version').split('.'))
                    # form the pyre happy name
                    name = cls.category + version
                    # hand back the pyre safe name and the pile of packages
                    yield name, packages

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Provide alternative compatible implementations of python on dpkg machines, starting
        with the package the user has selected as the default
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
        Identify the default implementation of postgres on macports machines
        """
        # on macports, postgres is a package group
        for package in packager.alternatives(group='postgresql'):
            # if the package name starts with 'postgresql'
            if package.startswith('postgresql'):
                # use the default implementation
                yield Default(name=package)

        # out of ideas
        return


# superclasses
from .ToolInstallation import ToolInstallation
from .LibraryInstallation import LibraryInstallation


# the implementation
class Default(
        ToolInstallation, LibraryInstallation,
        family='pyre.externals.postgres.default', implements=Postgres):
    """
    A generic postgres installation
    """

    # constants
    category = Postgres.category
    flavor = category

    # user configurable state
    psql = pyre.properties.str()
    psql.doc = 'the full path to the postgres client'

    defines = pyre.properties.strings(default="WITH_PQ")
    defines.doc = "the compile time markers that indicate my presence"

    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask dpkg for help; start by finding out which package supports me
        bin, dev = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=dev)

        # the name of the client
        self.psql = 'psql'
        # our search target for the bindir is in a bin directory to avoid spurious matches
        launcher = "bin/{.psql}".format(self)
        # find it in order to identify my {bindir}
        bindir = packager.findfirst(target=launcher, contents=packager.contents(package=bin))
        # and save it
        self.bindir = [ bindir / 'bin' ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = r'libpq-fe\.h'
        # find it
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = 'pq'
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


    def macports(self, packager, **kwds):
        """
        Attempt to repair my configuration
        """
        # ask macports for help; start by finding out which package supports me
        package = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=package)
        # and the package contents
        contents = tuple(packager.contents(package=package))

        # {postgresql} is a selection group
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
        # find the binary that supports {psql} and use it to set my launcher
        self.psql = nmap[pyre.primitives.path('bin/psql')].name
        # set my {bindir}
        bindir = packager.findfirst(target=self.psql, contents=contents)
        # and save it
        self.bindir = [ bindir ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = 'libpq-fe.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        libpq = self.pyre_host.dynamicLibrary('pq')
        # find it
        libdir = packager.findfirst(target=libpq, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = 'pq'

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.bindir+self.incdir+self.libdir)

        # all done
        return


# end of file
