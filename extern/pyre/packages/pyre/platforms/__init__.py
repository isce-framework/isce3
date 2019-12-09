# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the marker of component factories
from .. import foundry


# the protocols
from .Platform import Platform as platform
from .PackageManager import PackageManager as packager

# the various implementations
# package managers
@foundry(implements=packager)
def dpkg():
    """
    Support for the Debian packager
    """
    # get the class record
    from .DPkg import DPkg
    # and return it
    return DPkg


@foundry(implements=packager)
def macports():
    """
    Support for macports
    """
    # get the class record
    from .MacPorts import MacPorts
    # and return it
    return MacPorts


# host types
@foundry(implements=platform)
def darwin():
    """
    Support for OSX
    """
    # get the class record
    from .Darwin import Darwin
    # and return it
    return Darwin


@foundry(implements=platform)
def linux():
    """
    Generic support for linux flavors
    """
    # get the class record
    from .Linux import Linux
    # and return it
    return Linux


@foundry(implements=platform)
def centos():
    """
    Support for CentOS
    """
    # get the class record
    from .CentOS import CentOS
    # and return it
    return CentOS


@foundry(implements=platform)
def redhat():
    """
    Support for RedHat
    """
    # get the class record
    from .RedHat import RedHat
    # and return it
    return RedHat


@foundry(implements=platform)
def ubuntu():
    """
    Support for Ubuntu
    """
    # get the class record
    from .Ubuntu import Ubuntu
    # and return it
    return Ubuntu


# host aliases
osx = darwin


# end of file
