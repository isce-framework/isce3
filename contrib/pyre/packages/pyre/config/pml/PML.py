# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#



from ..Codec import Codec


class PML(Codec):
    """
    This package contains the implementation of the pml reader and writer
    """


    # constants
    encoding = "pml"


    # interface
    @classmethod
    def decode(cls, uri, source, locator):
        """
        Parse {source} and return its contents
        """
        # get access to the XML package
        import pyre.xml
        # and the pml document
        from .Document import Document
        # make a reader
        reader = pyre.xml.newReader()
        # parse the contents
        try:
            # harvest the configuration events in the source
            configuration = reader.read(stream=source, document=Document())
        except reader.ParsingError as error:
            # adjust the locator if necessary
            if locator:
                loc = pyre.tracking.chain(this=error.locator, next=locator)
            else:
                loc = error.locator
            msg = "decoding error: {}".format(error.description)
            # convert the parsing error into a decoding error and raise it
            raise cls.DecodingError(
                codec=cls, uri=uri, locator=loc, description=msg) from error
        # all done; return the harvested events
        # for event in configuration: print(event)
        return configuration


# end of file
