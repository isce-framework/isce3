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


# the python package manager
class Python(Tool, Library, family='pyre.externals.python'):
    """
    The package manager for the python interpreter
    """

    # constants
    category = 'python'

    # user configurable state
    interpreter = pyre.properties.str()
    interpreter.doc = 'the full path to the interpreter'


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
        rgx = r"^(?P<lib>lib(?P<base>python(?P<version>(?P<major>[0-9])\.(?P<minor>[0-9]))))-dev$"
        # the recognizer of python dev packages
        scanner = re.compile(rgx)

        # go through the names of all installed packages
        for key in installed.keys():
            # looking for ones that match my pattern
            match = scanner.match(key)
            # once we have a match
            if match:
                # extract the dev package
                dev = match.group()
                # the base package name
                base = match.group('base')
                # and the lib package
                libpython = match.group('lib')
                # form the minimal package name
                minimal = base + '-minimal'
                # put all the packages in a pile
                packages = base, minimal, libpython, dev
                # find the missing ones
                missing = [ pkg for pkg in packages if pkg not in installed ]
                # if there are no missing ones
                if not missing:
                    # extract the version and collapse it
                    version = ''.join(match.group('version').split('.'))
                    # form the pyre happy installation name
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
        versions = Python3, Python2
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
        Provide alternative compatible implementations of python on macports machines, starting
        with the package the user has selected as the default
        """
        # on macports, {python3} and {python2} are separate package groups; try python3.x
        # installations followed by python 2.x
        versions = [ Python3, Python2 ]
        # go through my choices
        for version in versions:
            # ask macports for all available alternatives
            for package in packager.alternatives(group=version.flavor):
                # instantiate each one using the package name and hand it to the caller
                yield version(name=package)

        # out of ideas
        return


# implementation superclasses
from .ToolInstallation import ToolInstallation
from .LibraryInstallation import LibraryInstallation


# the base class for python installations
class Default(
        ToolInstallation, LibraryInstallation,
        family='pyre.externals.python.default', implements=Python):
    """
    The base class for for python instances
    """

    # constants
    flavor = Python.category
    category = Python.category

    # public state
    libraries = pyre.properties.strings()
    libraries.doc = 'the libraries to place on the link line'

    interpreter = pyre.properties.str()
    interpreter.doc = 'the full path to the python interpreter'


    # configuration

    # these methods are invoked after construction if the instance is determined to be in
    # invalid state that was not the user's fault. typically this means that the package
    # configuration is still in its default state. the dispatcher determines the correct
    # package manager and forwards to one of the handlers in this section

    def dpkg(self, packager):
        """
        Attempt to repair my configuration
        """
        # ask dpkg for help; start by finding out which package supports me
        base, minimal, libpython, dev = packager.identify(installation=self)
        # get the version info
        self.version, _ = packager.info(package=base)

        # the name of the interpreter
        self.interpreter = '{0.category}{0.sigver}m'.format(self)
        # our search target for the bindir is in a bin directory to avoid spurious matches
        interpreter = "bin/{.interpreter}".format(self)
        # find it in order to identify my {bindir}
        bindir = packager.findfirst(target=interpreter, contents=packager.contents(package=minimal))
        # and save it
        self.bindir = [ bindir / 'bin' ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = 'Python.h'
        # find it
        incdir = packager.findfirst(target=header, contents=packager.contents(package=dev))
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = '{0.category}{0.sigver}m'.format(self)
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

        # the name of the interpreter
        self.interpreter = '{0.category}{0.sigver}m'.format(self)
        # find it in order to identify my {bindir}
        bindir = packager.findfirst(target=self.interpreter, contents=contents)
        # and save it
        self.bindir = [ bindir ] if bindir else []

        # in order to identify my {incdir}, search for the top-level header file
        header = 'Python.h'
        # find it
        incdir = packager.findfirst(target=header, contents=contents)
        # and save it
        self.incdir = [ incdir ] if incdir else []

        # in order to identify my {libdir}, search for one of my libraries
        stem = '{0.category}{0.sigver}m'.format(self)
        # convert it into the actual file name
        libpython = self.pyre_host.dynamicLibrary(stem)
        # find it
        libdir = packager.findfirst(target=libpython, contents=contents)
        # and save it
        self.libdir = [ libdir ] if libdir else []
        # set my library
        self.libraries = stem

        # now that we have everything, compute the prefix
        self.prefix = self.commonpath(folders=self.bindir+self.incdir+self.libdir)

        # all done
        return


# the python 2.x package manager
class Python2(Default, family='pyre.externals.python.python2'):
    """
    The package manager for python 2.x instances
    """

    # constants
    flavor = Default.flavor + '2'

    # public state
    defines = pyre.properties.strings(default="WITH_PYTHON2")
    defines.doc = "the compile time markers that indicate my presence"


# the python 3.x package manager
class Python3(Default, family='pyre.externals.python.python3'):
    """
    The package manager for python 3.x instances
    """

    # constants
    flavor = Default.flavor + '3'

    # public state
    defines = pyre.properties.strings(default="WITH_PYTHON3")
    defines.doc = "the compile time markers that indicate my presence"


# end of file
