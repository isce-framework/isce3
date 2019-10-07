# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Processor:
    """
    The base class for decorators that attach value processors to descriptors
    """


    # public data
    traits = () # the sequence of descriptors that i decorate


    # meta methods
    def __init__(self, traits=traits, **kwds):
        # chain up
        super().__init__(**kwds)
        # record which descriptors i decorate
        self.traits = tuple(traits)
        # all done
        return


    def __call__(self, method):
        raise NotImplementedError(
            "class {.__name__!r} must implement '__call__'".format(type(self)))


# end of file
