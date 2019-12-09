# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


from ..framework.exceptions import FrameworkError, BadResourceLocatorError


class ConfigurationError(FrameworkError):
    """
    Base class for all configuration errors
    """


class CodecError(ConfigurationError):
    """
    Base class for codec errors
    """

    # public data
    description = "generic codec error"

    # meta-methods
    def __init__(self, codec, uri="",  **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.codec = codec
        self.uri = uri
        # all done
        return


class UnknownEncodingError(CodecError):
    """
    A request for an unknown codec was made
    """

    # public data
    description = '{0.uri.uri!r}: unknown encoding {0.encoding!r}'

    # meta-methods
    def __init__(self, encoding, **kwds):
        # chain up
        super().__init__(codec=None, **kwds)
        # save the error info
        self.encoding = encoding
        # all done
        return


class DecodingError(CodecError):
    """
    Exception raised by codecs when they encounter errors in their input streams
    """


class EncodingError(CodecError):
    """
    Exception raised by codecs when they fail to inject an item in a stream
    """


class LoadingError(CodecError):
    """
    Exception raised by codecs when they encounter errors in their input streams
    """


class ShelfError(ConfigurationError):

    # meta-methods
    def __init__(self, shelf, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.shelf = shelf
        # all done
        return


class SymbolNotFoundError(ShelfError):

    # meta-methods
    description = "symbol {0.symbol!r} not found in {0.shelf!r}"

    # meta-methods
    def __init__(self, symbol, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the error info
        self.symbol = symbol
        # all done
        return


# end of file
