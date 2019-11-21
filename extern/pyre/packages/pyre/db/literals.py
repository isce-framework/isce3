# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# get the nodes
from ..calc.Node import Node as node


# the rich literals
class Literal(node.literal):

    # interface
    def sql(self):
        # easy enough
        return self.value

    # my representations
    def __str__(self): return self.value
    def __repr__(self): return self.value


# the constants
null = Literal(value='NULL')
default = Literal(value='DEFAULT')


# end of file
