# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my ancestors
from .Mill import Mill
from .LineComments import LineComments


# my declaration
class LineMill(Mill, LineComments):
    """
    A text generator for languages that have line oriented commenting
    """


# end of file
