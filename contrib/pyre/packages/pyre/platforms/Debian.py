# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework
import pyre
# superclass
from .Linux import Linux
# the default package manager
from .DPkg import DPkg


# declaration
class Debian(Linux, family='pyre.platforms.debian'):
    """
    Encapsulation of a host running linux on a debian derivative
    """

    # constants
    distribution = 'debian'


    # user configurable state
    packager = pyre.platforms.packager(default=DPkg)
    packager.doc = 'the manager of external packages installed on this host'


# end of file
