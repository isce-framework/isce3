# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
from .. import primitives


# declaration
class Naked:
    """
    A thin wrapper around local regular files
    """


    # types
    from .Stat import Stat as recognizer


    # constants
    isFolder = False


    # public data
    @property
    def marker(self):
        """
        Return my distinguishing mark used by explorers to decorate their reports
        """
        # easy enough
        return self._metadata.marker


    # interface
    def open(self, **kwds):
        """
        Open the file
        """
        # easy enough
        return self.uri.open(**kwds)


    # filesystem obligations
    def metadata(self, uri, **kwds):
        """
        Build a structure to hold the node metadata
        """
        # ignore the {uri} argument: it is just the path relative to the containing filesystem
        # ask my recognizer to do his job
        return self._metadata


    # meta-methods
    def __init__(self, uri, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my name
        self.uri = primitives.path(uri).resolve()
        # build my metadata
        self._metadata = self.recognizer.recognize(uri)
        # all done
        return


# end of file
