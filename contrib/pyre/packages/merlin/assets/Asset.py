# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Asset:
    """
    Base class for all objects tracked by merlin
    """


    # constants
    category = "asset"


    # meta methods
    def __init__(self, name, uri, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my properties
        self.name = name # my name
        self.uri = uri # my path relative to the top level container
        # all done
        return


    # implementation details
    __slots__ = 'name', 'uri'


    # debugging support
    def dump(self, indent=''):
        print('{0}{1.name} ({1.category})'.format(indent, self))
        return


# end of file
