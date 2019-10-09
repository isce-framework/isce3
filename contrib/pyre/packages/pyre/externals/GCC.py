# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import re, subprocess
# access to the framework
import pyre
# superclass
from .Tool import Tool


# the gcc package manager
class GCC(Tool, family='pyre.externals.gcc'):
    """
    The package manager for GCC installations
    """

    # constants
    category = 'gcc'

    # public state
    wrapper = pyre.properties.str(default='gcc')
    wrapper.doc = "the name of the compiler front end"


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
        rgx = r"^gcc-(?P<version>(?P<major>[0-9]+)(\.(?P<minor>[0-9]+))?)$"
        # the recognizer of python dev packages
        scanner = re.compile(rgx)

        # go through the names of all installed packages
        for key in installed.keys():
            # looking for ones that match my pattern
            match = scanner.match(key)
            # once we have a match
            if match:
                # extract the version and collapse it
                version = ''.join(match.group('version').split('.'))
                # form the pyre happy name
                name = 'gcc' + version
                # and the sequence of packages
                packages = match.group(),
                # hand them to the caller
                yield name, packages

        # all done
        return


    @classmethod
    def dpkgPackages(cls, packager):
        """
        Provide alternative compatible implementations of python on dpkg machines, starting
        with the package the user has selected as the default
        """
        # ask dpkg for the index of alternatives
        alternatives = sorted(packager.alternatives(group=cls), reverse=True)
        # the supported versions in order of preference
        versions = GCC5, GCC4
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
        Identify the default implementation of GCC on macports machines
        """
        # the list of supported versions
        versions = [ GCC5 ]
        # go through my choices
        for version in versions:
            # ask macports for all available alternatives
            for package in packager.alternatives(group=version.flavor):
                # instantiate each one using the package name and hand it to the caller
                yield version(name=package)

        # out of ideas
        return


# superclass
from .ToolInstallation import ToolInstallation


# the implementation of a GCC installation
class Default(ToolInstallation, family='pyre.externals.gcc.gcc', implements=GCC):
    """
    Support for GCC installations
    """

    # constants
    category = GCC.category
    flavor = category

    # public state
    wrapper = pyre.properties.str()
    wrapper.doc = "the name of the compiler front end"


    # configuration
    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask dpkg for help; start by finding out which package supports me
        gcc, *_ = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=gcc)

        # get my flavor
        flavor = self.flavor
        # set the name of the compiler
        self.wrapper = gcc
        # the search target specifies a {bin} directory to avoid spurious matches
        wrapper = 'bin/{.wrapper}'.format(self)
        # find it in order to identify my {bindir}
        prefix = packager.findfirst(target=wrapper, contents=packager.contents(package=gcc))
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
        # ask macports for help; start by finding out which package supports me
        package = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=package)
        # and the package contents
        contents = tuple(packager.contents(package=package))

        # {gcc} is a selection group
        group = self.category
        # the package deposits its selection alternative here
        selection = str(packager.prefix() / 'etc' / 'select' / group / '(?P<alternate>.*)')
        # so find it
        match = next(packager.find(target=selection, pile=contents))
        # extract the name of the alternative
        alternative = match.group('alternate')
        # ask for the normalization data
        normalization = packager.getNormalization(group=group, alternative=alternative)
        # build the map
        nmap = { base: target for base,target in zip(*normalization) }
        # find the binary that supports {gcc} and use it to set my wrapper
        self.wrapper = nmap[pyre.primitives.path('bin/gcc')].name
        # look for it to get my {bindir}
        bindir = packager.findfirst(target=self.wrapper, contents=contents)
        # and save it
        self.bindir = [ bindir ] if bindir else []

        # now that we have everything, compute the prefix
        self.prefix = self.bindir[0].parent

        # all done
        return


    # implementation details
    def retrieveVersion(self):
        """
        Get my version number directly from the compiler

        In general, this is not needed except on hosts with no package managers to help me
        """
        # set up the shell command
        settings = {
            'executable': self.wrapper,
            'args': (self.wrapper, '--version'),
            'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': False
        }
        # make a pipe
        with subprocess.Popen(**settings) as pipe:
            # get the text source
            stream = pipe.stdout

            # the first line is the version
            line = next(stream).strip()
            # extract the fields
            match = self._versionRegex.match(line)
            # if it didn't match
            if not match:
                # oh well...
                return 'unknown'
            # otherwise, extract the clang version number
            return match.group('version')

        # all done
        return 'unknown'


    # private data
    _versionRegex = re.compile(r"gcc\s+\([^)]+\)\s+(?P<version>[.0-9]+)")


# specific versions
class GCC4(Default, family='pyre.externals.gcc.gcc4'):
    """
    Support for GCC 4.x installations
    """

    # constants
    flavor = Default.category + '4'


class GCC5(Default, family='pyre.externals.gcc.gcc5'):
    """
    Support for GCC 5.x installations
    """

    # constants
    flavor = Default.category + '5'


# Apple's clang
class CLang(ToolInstallation, family='pyre.externals.gcc.clang', implements=GCC):
    """
    Apple's clang
    """

    # constants
    category = GCC.category


    # public state
    wrapper = pyre.properties.str(default='/usr/bin/gcc')
    wrapper.doc = "the name of the compiler front end"


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # get the version
        self.version = self.retrieveVersion()
        # all done
        return


    # implementation details
    def retrieveVersion(self):
        """
        Get my version number
        """
        # set up the shell command
        settings = {
            'executable': self.wrapper,
            'args': (self.wrapper, '--version'),
            'stdout': subprocess.PIPE, 'stderr': subprocess.PIPE,
            'universal_newlines': True,
            'shell': False
        }
        # make a pipe
        with subprocess.Popen(**settings) as pipe:
            # get the text source
            stream = pipe.stdout

            # the first line is the version
            line = next(stream).strip()
            # extract the fields
            match = self._versionRegex.match(line)
            # if it didn't match
            if not match:
                # oh well...
                return 'unknown'
            # otherwise, extract the clang version number
            return match.group('version')

        # all done
        return


    # private data
    _versionRegex = re.compile(r"Apple LLVM version [.0-9]+\s\(clang-(?P<version>[.0-9]+)\)$")


# end of file
