# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Chain:
    """
    A locator that ties together two others in order to express that something in {next}
    caused {this} to be recorded
    """


    # meta methods
    def __init__(self, this, next):
        self.this = this
        self.next = next
        return


    def __str__(self):
        # if {next} is non-trivial, show the chain
        if self.next: return "{0.this}, {0.next}".format(self)
        # otherwise don't
        return "{0.this}".format(self)


    # implementation details
    __slots__ = "this", "next"


# end of file
