# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Debian import Debian


# declaration
class Ubuntu(Debian, family='pyre.platforms.ubuntu'):
    """
    Encapsulation of a host running linux on the ubuntu distribution
    """

    # constants
    distribution = 'ubuntu'


# end of file
