# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# built-in spells
from .. import foundry, action


# meta activities
@foundry(implements=action, tip="create a new merlin project")
def init():
    """
    Create a new merlin project
    """
    from .Initializer import Initializer
    return Initializer


@foundry(implements=action, tip="add the contents of the current directory to the project")
def add():
    """
    Add the contents of the current directory to the project
    """
    from .AssetManager import AssetManager
    return AssetManager


# administrivia
@foundry(implements=action, tip="display information about the current machine, user and project")
def about():
    """
    Display information about the current machine, user and project
    """
    from .About import About
    return About


# end of file
