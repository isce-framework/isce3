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
class Make(LineMill):
    """
    Support for makefiles
    """


    # traits
    languageMarker = pyre.properties.str(default='Makefile')
    languageMarker.doc = "the language marker"


    # private data
    comment = '#'


# end of file
