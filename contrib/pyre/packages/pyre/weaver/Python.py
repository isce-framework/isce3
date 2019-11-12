# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestors
from .LineMill import LineMill
from .Expression import Expression


# my declaration
class Python(LineMill, Expression):
    """
    Support for python
    """


    # traits
    version = pyre.properties.str(default='3')
    version.doc = "the version of python to use on the hash-bang line"

    languageMarker = pyre.properties.str(default='Python')
    languageMarker.doc = "the language marker"

    script = pyre.properties.bool(default=False)
    script.doc = "controls whether to render a hash-bang line appropriate for script files"


    # interface
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # if this is an executable script
        if self.script:
            # render the hash-bang line
            yield "#!/usr/bin/env python" + self.version
        # and the rest
        yield from super().header()
        # all done
        return


    # private data
    comment = '#'


# end of file
