# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# super class
from .Asset import Asset


# class declaration
class PythonModule(Asset):
    """
    Encapsulation of an asset that represents a python source file
    """


    # constants
    category = "python module"


    # implementation details
    __slots__ = ()


# end of file
