# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .AssetContainer import AssetContainer


# class declaration
class PythonPackage(AssetContainer):
    """
    Encapsulation of a python package, i.e. a folder of python modules
    """


    # constants
    category = "python package"


    # implementation details
    __slots__ = ()


# end of file
