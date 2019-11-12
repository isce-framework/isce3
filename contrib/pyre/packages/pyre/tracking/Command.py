# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Command:
    """
    A locator that records the position of a command line argument
    """


    # constant
    source = 'from the command line argument {!r}'


    # meta methods
    def __init__(self, arg):
        self.arg = arg
        return


    def __str__(self):
        return self.source.format(self.arg)


    # implementation details
    __slots__ = 'arg',


# end of file
