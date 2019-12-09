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
class Perl(LineMill):
    """
    Support for perl
    """


    # traits
    version = pyre.properties.str(default='5')
    version.doc = "the version of perl to use on the hash-bang line"


    # interface
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # if i have been given an explicit version number
        if self.version:
            # use it
            yield "#!/usr/bin/env perl" + self.version
        # otherwise
        else:
            # render a generic hash-bang line
            yield "#!/usr/bin/env perl"
        # and the rest
        yield from super().header()
        # all done
        return


    # private data
    comment = '#'


# end of file
