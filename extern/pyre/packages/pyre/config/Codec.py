# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Codec:
    """
    The base class for readers/writers of the pyre configuration files
    """


    # types
    # exceptions
    from .exceptions import EncodingError, DecodingError, ShelfError, SymbolNotFoundError


    # public data: descendants must specify these
    encoding = None


    # abstract interface
    @classmethod
    def encode(self, item, stream):
        """
        Build a representation of {item} in the current encoding and inject it into {stream}
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'encode'".format(type(self)))


    @classmethod
    def decode(self, client, scheme, source, locator=None):
        """
        Ingest {source} and return the decoded contents
        """
        raise NotImplementedError(
            "class {.__name__!r} must override 'decode'".format(type(self)))


# end of file
