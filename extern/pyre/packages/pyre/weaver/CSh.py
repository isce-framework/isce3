# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestor
from .LineMill import LineMill


# my declaration
class CSh(LineMill):
    """
    Support for the c shell
    """


    # traits
    variant = pyre.properties.str(default='/bin/csh')
    variant.doc = "the shell variant to use on the hash-bang line"


    # interface
    @pyre.export
    def header(self):
        """
        Render a hash-bang line if i know the shell variant
        """
        # render the hash-bang line
        if self.variant: yield "#!" + self.variant
        # and the rest
        yield from super().header()
        # all done
        return


    # private data
    comment = '#'


# end of file
