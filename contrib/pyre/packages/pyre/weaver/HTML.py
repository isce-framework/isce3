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
class HTML(BlockMill):
    """
    Support for HTML
    """


    # traits
    doctype = pyre.properties.str(default='html5')
    doctype.doc = "the doctype variant to use on the first line"


    # mill obligations
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header
        """
        # if I have doctype
        if self.doctype:
            # translate and render it
            yield "<!doctype html{}>".format(self.doctypes[self.doctype])
        # render the rest
        yield from super().header()
        # the outer tag
        yield '<html>'
        # all done
        return


    @pyre.export
    def body(self, document=(), **kwds):
        """
        The body of the document
        """
        # if the document is a simple string
        if isinstance(document, str):
            # the caller has done all the rendering; just return it
            yield document
            # all done
            return

        # if it is some kind of exception
        if isinstance(document, Exception):
            # convert it into a string
            yield str(document)
            # all done
            return

        # otherwise, chain up
        yield from super().body(document=document, **kwds)


    @pyre.export
    def footer(self):
        """
        Layout the {document} using my stationery for the footer
        """
        # close the top level tag
        yield '</html>'
        # chain up
        yield from super().footer()
        # all done
        return


    # constants
    doctypes = {
        'html5': '',
        'html4-strict':
            ' public "-//w3c//dtd html 4.01//en" "http://www.w3.org/TR/html4/strict.dtd"',
        'html4-transitional': ' public "-//w3c//dtd html 4.01 transitional//en"',
        }


    # private data
    startBlock = '<!--'
    commentMarker = ' !'
    endBlock = '-->'


# end of file
