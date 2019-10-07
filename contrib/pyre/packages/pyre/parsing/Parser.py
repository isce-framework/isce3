# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Parser:
    """
    The base class for parsers
    """


    # types
    from .exceptions import ParsingError, SyntaxError, TokenizationError


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # build my scanner
        self.scanner = self.lexer()
        # all done
        return


    # implementation details
    lexer = None # my scanner factory
    scanner = None # my scanner instance


# end of file
