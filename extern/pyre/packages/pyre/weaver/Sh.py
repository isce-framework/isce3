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
class Sh(LineMill):
    """
    Support for the Bourne shell
    """


    # traits
    variant = pyre.properties.str(default='/bin/bash')
    variant.doc = "the shell variant to use on the hash-bang line"


    # interface
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # if I know which shell I am
        if self.variant:
            # render the hash-bang line
            yield "#!" + self.variant
        # and the rest
        yield from super().header()
        # all done
        return


    # private data
    comment = '#'


# end of file
