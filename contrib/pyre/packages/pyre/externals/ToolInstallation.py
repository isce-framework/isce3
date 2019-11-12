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


# the base installation manager for tools
class ToolInstallation(Installation):
    """
    The package manager for generic tools
    """

    # public state
    bindir = pyre.properties.paths()
    bindir.doc = "the location of my binaries"


    # protocol obligations
    @pyre.export
    def binaries(self, **kwds):
        """
        A sequence of names of binaries to look for
        """
        # must have one
        return ()


    # framework hooks
    def pyre_configured(self):
        """
        Verify that the {bindir} trait points to a good location
        """
        # chain up
        yield from super().pyre_configured()
        # check that my {bindir} exists
        yield from self.verify(trait='bindir', folders=self.bindir)

        # all done
        return


# end of file
