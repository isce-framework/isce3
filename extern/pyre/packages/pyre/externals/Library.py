# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre
# superclass
from .Package import Package


# my declaration
class Library(Package):
    """
    Base class for third party libraries
    """

    # user configurable state
    defines = pyre.properties.strings()
    defines.doc = "the compile time markers that indicate my presence"

    incdir = pyre.properties.paths()
    incdir.doc = "the locations of my headers; for the compiler command line"

    libdir = pyre.properties.paths()
    libdir.doc = "the locations of my libraries; for the linker command path"


# end of file
