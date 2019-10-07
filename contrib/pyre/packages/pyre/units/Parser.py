# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from ..patterns.Singleton import Singleton


class Parser(metaclass=Singleton):
    """
    Singleton that converts string representations of dimensional quantities into instances of
    Dimensional
    """


    # interface
    def parse(self, text, context=None):
        """
        Convert the string representation in {text} into a dimensional quantity
        """
        # check for "none"
        if text.strip().lower() == "none":
            # do as told
            return None

        # if the caller did not supply a context
        if context is None:
            # use ours
            context = self.context
        # otherwise
        else:
            # create a new one
            context = context.copy()
            # and merge mine in it
            context.update(self.context)

        # evaluate the expression and return the result
        return eval(text, context)


    # meta methods
    def __init__(self, **kwds):
        super().__init__(**kwds)
        self.context = self._initializeContext()
        return


    # implementation details
    def _initializeContext(self):
        """
        Build the initial list of resolvable unit symbols
        """
        # get the list of default packages
        from . import quantities
        # initialize the pile
        context = {}
        # go through all the supported quantities
        for quantity in quantities():
            # extract the symbols and their value
            for symbol, value in quantity.__dict__.items():
                # skip symbols that are in the reserved python namespace
                if not symbol.startswith('__'):
                    # add the rest to the pile
                    context[symbol] = value
        # and return it
        return context


    # access to the dimensional factory
    from . import dimensional


# end of file
