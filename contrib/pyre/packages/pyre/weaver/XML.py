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
class XML(BlockMill):
    """
    Support for XML
    """


    # mill obligations
    @pyre.export
    def header(self):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # render the xml marker
        yield '<?xml version="1.0"?>'
        # and the rest
        yield from super().header()
        # all done
        return


    # interface
    def push(self, tag, attributes=None):
        """
        Open a {tag}
        """
        # if I have non-trivial attributes
        if attributes is not None:
            # render them
            attr = ''.join(' {}="{}"'.format(key, value) for key,value in attributes.items())
        # otherwise
        else:
            # boring...
            attr = ''
        # build the tag and send it
        yield '{.leader}<{}{}>'.format(self, tag, attr)
        # add the {tag} to the pile
        self.tags.append(tag)
        # indent
        self.indent()
        # all done
        return


    def pop(self):
        """
        Pop a tag from the stack
        """
        # outdent
        self.outdent()
        # close the tag
        yield '{.leader}</{}>'.format(self, self.tags.pop())
        # all done
        return


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my tag stack
        self.tags = []
        # all done
        return


    # private data
    startBlock = '<!--'
    commentMarker = ' !'
    endBlock = '-->'


# end of file
