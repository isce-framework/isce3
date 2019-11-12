# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my superclass
from ..Codec import Codec


# class declaration
class Config(Codec):
    """
    This package contains the implementation of the {cfg} reader and writer
    """


    # constants
    encoding = "cfg"


    # interface
    @classmethod
    def decode(cls, uri, source, locator):
        """
        Parse {source} and return the configuration events it contains
        """
        # get the parser factory
        from .Parser import Parser
        # make a parser
        parser = Parser()
        # harvest the configuration events
        configuration = parser.parse(uri=uri, stream=source, locator=locator)
        # grab the accumulated errors
        errors = parser.errors
        # if there were no errors
        if not errors:
            # return the harvested configuration events
            return configuration
        # otherwise, throw away the harvested events
        return []



# end of file
