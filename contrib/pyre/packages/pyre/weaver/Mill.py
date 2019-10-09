# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre


# my ancestors
from .Indenter import Indenter
# my protocol
from .Language import Language


# turn my subclasses into components
class Mill(pyre.component, Indenter, implements=Language):
    """
    The base class for text renderers
    """


    # types
    # the protocols of my traits
    from .Stationery import Stationery


    # traits
    stationery = Stationery()
    stationery.doc = "the overall layout of the document"

    languageMarker = pyre.properties.str()
    languageMarker.doc = "the string to use as the language marker"


    # interface
    @pyre.export
    def render(self, **kwds):
        """
        Layout the {document} using my stationery for the header and footer
        """
        # create the header
        yield from self.header()
        # and a blank line
        yield ''

        # process it
        yield from self.body(**kwds)
        # another blank line
        yield ''

        # and the footer
        yield from self.footer()
        # all done
        return


    # the lower level interface
    @pyre.export
    def header(self):
        """
        Build the header of the document
        """
        # the low level guy does all the work; just wrap everything in a comment block
        yield from self.commentBlock(self._header())
        # all done
        return


    @pyre.export
    def body(self, document=()):
        """
        The body of the document
        """
        # empty, by default, but maybe the caller has already built one. of course, the proper
        # way is to have a subclass do something smart, but some users may want to dump into
        # the file text they have already prepared
        yield from document


    @pyre.export
    def footer(self):
        """
        Build the footer of the document
        """
        # cache my stationery
        stationery = self.stationery
        # if we have a footer
        if stationery.footer:
            # render the footer
            yield self.commentLine(stationery.footer)
        # all done
        return


    # implementation details
    def _header(self):
        """
        Workhorse for the header generator
        """
        # cache my stationery
        stationery = self.stationery
        # if we have a language marker
        if self.languageMarker:
            # render it
            yield "-*- {.languageMarker} -*-" .format(self)
        # a blank, commented line
        yield ''
        # render the authors
        yield from stationery.authors
        # render the affiliation
        if stationery.affiliation:
            yield stationery.affiliation
        # render the copyright note
        if stationery.copyright:
            yield stationery.copyright
        # a blank, commented line
        yield ''
        # all done
        return


# end of file
