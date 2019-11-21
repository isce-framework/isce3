# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestor
from .BlockMill import BlockMill


# my declaration
class SVG(BlockMill):
    """
    Support for SVG, the scalable vector graphics format
    """


    # user configurable state
    standalone = pyre.properties.bool(default=True)


    # interface
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # if this is a stand-alone document
        if self.standalone:
            # render the xml marker
            yield '<?xml version="1.0"?>'
            # the document header
            yield from super().header()
            # and a blank line
            yield ''
        # render the svg tag
        yield '<svg version="1.1" xmlns="http://www.w3.org/2000/svg">'
        # all done
        return


    @pyre.export
    def footer(self):
        """
        Build the document footer
        """
        # close the svg tag
        yield '</svg>'
        # if this is a stand-alone document
        if self.standalone:
            # render a blank line
            yield ''
            # and the document footer
            yield from super().footer()
        # all done
        return


    # private data
    startBlock = '<!--'
    commentMarker = ' !'
    endBlock = '-->'


# end of file
