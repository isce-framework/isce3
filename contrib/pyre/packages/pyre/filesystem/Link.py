# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .File import File


# class declaration
class Link(File):
    """
    Representation of symbolic links for filesystems that support them
    """


    # constant
    marker = 'l'


    # interface
    def identify(self, explorer, **kwds):
        """
        Guide {explorer}
        """
        return explorer.onLink(info=self, **kwds)


    # meta-methods
    def __init__(self, uri, info=None, **kwds):
        # chain up
        super().__init__(uri=uri, info=info, **kwds)
        # support
        from .Stat import Stat
        # build my referent
        self.referent = Stat.recognize(entry=uri, follow_symlinks=True)
        # all done
        return


# end of file
