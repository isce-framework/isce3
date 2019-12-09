# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .InfoFile import InfoFile
from .InfoZip import InfoZip


# class declaration
class InfoZipFile(InfoZip, InfoFile):
    """
    The base class for zip filesystem entries
    """


# end of file
