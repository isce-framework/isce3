# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclasses
from .Observer import Observer
from .Observable import Observable


# declaration
class Probe(Observer, Observable):
    """
    The base class for entities that observe the values of nodes in a calc graph and can notify
    external clients when node values change
    """


# end of file
