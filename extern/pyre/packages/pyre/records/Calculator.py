# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Calculator:
    """
    A strategy for pulling data from a stream that attaches the values to {pyre.calc} nodes,
    making the record instances mutable
    """

    # types
    from ..calc.Node import Node as node


    # meta-methods
    def __call__(self, record, source, **kwds):
        """
        Pull values from {source}, convert and yield them
        """
        # make {pyre.calc} variables so record fields are mutable
        variable = self.node.variable
        # zip together the data stream and the descriptors
        for field, value in zip(record.pyre_fields, source):
            # build a node to hold it
            node = variable(value=value, postprocessor=field.process)
            # and make it available
            yield node
        # all done
        return


# end of file
