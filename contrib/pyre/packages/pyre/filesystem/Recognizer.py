# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Recognizer:
    """
    Abstract base class for filesystem entry recognition
    """


    # interface
    def recognize(self, entry):
        """
        Given a filesystem {entry}, build a filesystem specific structure and decorate it with
        the available metadata
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'recognize'".format(type(self)))


# end of file
