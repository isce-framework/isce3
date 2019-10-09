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


# the cython package manager
class Cython(Tool, family='pyre.externals.cython'):
    """
    The package manager for the cython interpreter
    """

    # constants
    category = 'cython'

    # user configurable state
    compiler = pyre.properties.str()
    compiler.doc = 'the name of the compiler; may be the full path to the executable'


    # support for specific package managers
    @classmethod
    def dpkgAlternatives(cls, dpkg):
        """
        Go through the installed packages and identify those that are relevant for providing
        support for my installations
        """
        # get the index of installed packages
        installed = dpkg.installed()

        # the cython 3.x development packages
        cython3 = 'cython3',
        # find the missing ones
        missing = [ pkg for pkg in cython3 if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield Cython3.flavor, cython3

        # the cython 2.x development packages
        cython2 = 'cython',
        # find the missing ones
        missing = [ pkg for pkg in cython2 if pkg not in installed ]
        # if there are no missing ones
        if not missing:
            # hand back a pyre safe name and the list of packages
            yield Cython2.flavor, cython2

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
        versions = Cython3, Cython2
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
        Provide alternative compatible implementations of cython on macports machines, starting
        with the package the user has selected as the default
        """
        # on macports, {cython3} and {cython2} are in the same package group; try cython3.x
        # installations followed by cython 2.x
        versions = [ Cython3, Cython2 ]
        # go through my choices
        for version in versions:
            # ask macports for all available alternatives
            for package in packager.alternatives(group=version.category):
                # instantiate each one using the package name and hand it to the caller
                yield version(name=package)

        # out of ideas
        return


# superclass
from .ToolInstallation import ToolInstallation


# the cython package manager
class Default(
        ToolInstallation,
        family='pyre.externals.cython.default', implements=Cython):
    """
    The package manager for cython instances
    """

    # constants
    category = Cython.category
    flavor = category

    # public state
    compiler = pyre.properties.str()
    compiler.doc = 'the name of the cython compiler'


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # get the names of the packages that support me
        bin, *_ = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=bin)

        # get my flavor
        flavor = self.flavor
        # the name of the interpreter
        self.compiler = 'cython' if flavor == 'cython2' else flavor
        # look for it in the bin directory so we don't pick up something else
        compiler = 'bin/{}'.format(self.compiler)
        # find it in order to identify my {bindir}
        prefix = packager.findfirst(target=compiler, contents=packager.contents(package=bin))
        # and save it
        self.bindir = [ prefix / 'bin' ] if prefix else []

        # set the prefix
        self.prefix = prefix

        # all done
        return


    def macports(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask macports for help; start by finding out which package is related to me
        package = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=package)
        # and the package contents
        contents = tuple(packager.contents(package=package))

        # {cython} is a selection group
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
        # find the binary that supports {cython} and use it to set my compiler
        self.compiler = nmap[pyre.primitives.path('bin/cython')].name
        # look for it to get my {bindir}
        bindir = packager.findfirst(target=self.compiler, contents=contents)
        # and save it
        self.bindir = [ bindir ] if bindir else []

        # now that we have everything, compute the prefix
        self.prefix = self.bindir[0].parent

        # all done
        return


# cython 2
class Cython2(Default, family='pyre.externals.cython.cython2'):
    """
    The package manager for cython 2.x instances
    """

    # constants
    flavor = Default.flavor + '2'


# cython 3
class Cython3(Default, family='pyre.externals.cython.cython3'):
    """
    The package manager for cython 3.x instances
    """

    # constants
    flavor = Default.flavor + '3'


# end of file
