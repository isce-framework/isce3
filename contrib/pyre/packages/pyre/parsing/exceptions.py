# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for the exceptions raised by this package
"""


# exceptions
from ..framework.exceptions import FrameworkError


class ParsingError(FrameworkError):
    """
    Base class for parsing exceptions

    Can be used to catch all exceptions raised by this package
    """


class IndentationError(ParsingError):
    """
    Exception raised when the scanner detects an inconsistent indentation pattern in the source
    """

    # public data
    description = "bad indentation"


class SyntaxError(ParsingError):
    """
    Exception raised when a syntax error is detected
    """
    # public data
    description = "syntax error: {0.token.lexeme!r}"

    # meta-methods
    def __init__(self, token, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the out of place token
        self.token = token
        # all done
        return


class TokenizationError(ParsingError):
    """
    Exception raised when the scanner fails to extract a token from the input stream
    """

    # public data
    description = "could not match {0.text!r}"

    # meta-methods
    def __init__(self, text, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the text we couldn't tokenize
        self.text = text
        # all done
        return


# end of file
