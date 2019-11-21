# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access the framework
import pyre
# and the merlin singletons
from .Dashboard import Dashboard as dashboard


# declaration
class Action(pyre.action, dashboard, family="merlin.spells"):
    """
    Protocol declaration for merlin spells
    """


# end of file
