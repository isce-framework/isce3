# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre


# declaration
class PackageManager(pyre.protocol, family='pyre.platforms.packagers'):
    """
    Encapsulation of host specific information
    """


    # requirements
    @pyre.provides
    def prefix(self):
        """
        The package manager install location
        """

    @pyre.provides
    def installed(self):
        """
        Retrieve available information for all installed packages
        """

    @pyre.provides
    def packages(self, category):
        """
        Provide a sequence of package names that provide compatible installations for the given
        package {category}. If the package manager provides a way for the user to select a
        specific installation as the default, care should be taken to rank the sequence
        appropriately.
        """

    @pyre.provides
    def info(self, package):
        """
        Return information about the given {package}

        The type of information returned is determined by the package manager. This method
        should return success if and only if {package} is actually fully installed.
        """

    @pyre.provides
    def contents(self, package):
        """
        Generate a sequence of the contents of {package}

        The type of information returned is determined by the package manager. Typically, it
        contains the list of files that are installed by this package, but it may contain other
        filesystem entities as well. This method should return a non-empty sequence if and only
        if {pakage} is actually fully installed
        """

    @pyre.provides
    def configure(self, packageInstance):
        """
        Dispatch to the {packageInstance} configuration procedure that is specific to the
        particular implementation of this protocol
        """


    # framework obligations
    @classmethod
    def pyre_default(cls, **kwds):
        """
        Build the preferred host implementation
        """
        # the host should specify a sensible default; if there is nothing there, this is an
        # unmanaged system that relies on environment variables and standard locations
        from .Bare import Bare
        # return the support for unmanaged systems
        return Bare


# end of file
