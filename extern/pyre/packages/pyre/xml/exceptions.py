# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


from . import newLocator
from ..framework.exceptions import FrameworkError


class ParsingError(FrameworkError):
    """
    Base class for parsing errors
    """

    # meta-methods
    def __init__(self, parser=None, document=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.parser = parser
        self.document = document
        # all done
        return


class UnsupportedFeatureError(ParsingError):
    """
    Exception raised when one of the requested features is not supported by the parser
    """

    # meta-methods
    def __init__(self, features, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.features = features
        self.description = "unsupported features: {0!r}".format(", ".join(features))
        # all done
        return


class DTDError(ParsingError):
    """
    Errors relating to the structure of the document
    """


class ProcessingError(ParsingError):
    """
    Errors relating to the handling of the document
    """

    # public data
    description = "unknown processing error"

    # meta-methods
    def __init__(self, saxlocator, **kwds):
        # convert the SAX locator to one of our own
        locator = newLocator(saxlocator) if saxlocator else None
        # chain up
        super().__init__(locator=locator, **kwds)
        # all done
        return


# end of file
