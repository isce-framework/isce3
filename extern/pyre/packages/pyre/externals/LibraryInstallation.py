# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework
import pyre
# superclass
from .Installation import Installation


# the base installation manager for libraries
class LibraryInstallation(Installation):
    """
    The package manager for libraries
    """

    # public state
    defines = pyre.properties.strings()
    defines.doc = "the compile time markers that indicate my presence"

    incdir = pyre.properties.paths()
    incdir.doc = "the locations of my headers; for the compiler command line"

    libdir = pyre.properties.paths()
    libdir.doc = "the locations of my libraries; for the linker command path"


    # framework hooks
    def pyre_configured(self):
        """
        Verify that my {incdir} and {libdir} traits point to good locations
        """
        # chain up
        yield from super().pyre_configured()
        # check that my {incdir} exists
        yield from self.verify(trait='incdir', folders=self.incdir)
        # check that my {libdir} exists
        yield from self.verify(trait='libdir', folders=self.libdir)

        # all done
        return


# end of file
