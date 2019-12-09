# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
from .. import actor, foundry


# declaration
class Shelf(dict):
    """
    Shelves are symbol tables that map component record factories to their names.
    """


    # exceptions
    from .exceptions import SymbolNotFoundError


    # public data
    uri = None
    locator = None


    # interface
    def retrieveSymbol(self, symbol):
        """
        Retrieve {symbol} from this shelf
        """
        # attempt to
        try:
            # look up the symbol
            return self[symbol]
        # if that fails
        except KeyError as error:
            # convert the error into something more descriptive and complain
            raise self.SymbolNotFoundError(shelf=self, symbol=symbol) from error

        # unreachable
        import journal
        raise journal.firewall('pyre.config.native').log("UNREACHABLE")


    # meta methods
    def __init__(self, uri, symbols, locator=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my state
        self.uri = uri
        self.locator = locator

        # go through all the loaded symbols
        for symbol, entity in symbols:
            # look for a foundry or a component class
            if isinstance(entity, foundry) or isinstance(entity, actor):
                # and save it
                self[symbol] = entity

        # ready to go
        return


# end of file
