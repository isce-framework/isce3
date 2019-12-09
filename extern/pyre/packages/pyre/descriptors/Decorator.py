# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my mix-ins
from .Typed import Typed
from .Public import Public
# superclass
from .. import algebraic


# declaration
class Decorator(algebraic.algebra):
    """
    Metaclass that decorates descriptors with a name and a type
    """


    # constants
    decorations = (Typed, Public)


    # framework support
    @classmethod
    def variableDerivation(cls, record):
        """
        Inject the local decorations to the variable inheritance hierarchy
        """
        # my local decorations
        yield from cls.decorations
        # plus whatever else my ancestors have to say
        yield from super().variableDerivation(record)
        # all done
        return


    @classmethod
    def operatorDerivation(cls, record):
        """
        Inject the local decorations to the operator inheritance hierarcrhy
        """
        # my local decorations
        yield from cls.decorations
        # plus whatever my ancestors have to say
        yield from super().operatorDerivation(record)
        # all done
        return


# end of file
