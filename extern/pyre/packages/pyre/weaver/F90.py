# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestor
from .LineMill import LineMill


# my declaration
class F90(LineMill):
    """
    Support for Fortran 90
    """


    # traits
    languageMarker = pyre.properties.str(default='F90')
    languageMarker.doc = "the variant to use in the language marker"


    # private data
    comment = '!'


# end of file
