# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my base class
from ..Shelf import Shelf as base


# declaration
class Shelf(base):
    """
    A symbol table primed by extracting symbols from a native python module
    """

    # N.B: shelves used to build their locators using the {__file__} attribute of modules;
    # unfortunately, this attribute is not set for all modules on all platforms, so it is not
    # reliable. this version just records the caller's locator


    # exceptions
    from ..exceptions import SymbolNotFoundError


    # meta methods
    def __init__(self, module=None, **kwds):
        # no module, no symbols
        symbols = module.__dict__.items() if module else ()
        # chain up
        super().__init__(symbols=symbols, **kwds)
        # all done
        return


# end of file
