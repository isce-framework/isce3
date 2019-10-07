# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .InfoFile import InfoFile
from .InfoStat import InfoStat


# class declaration
class File(InfoStat, InfoFile):
    """
    The base class for local filesystem entries
    """


    # interface
    def chmod(self, permissions):
        """
        Apply {permissions} to this file
        """
        # just do it...
        return self.uri.chmod(permissions)


# end of file
