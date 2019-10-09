# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my ancestors
from .Mill import Mill
from .BlockComments import BlockComments


# my declaration
class BlockMill(Mill, BlockComments):
    """
    A text generator for languages that have block oriented commenting
    """


# end of file
