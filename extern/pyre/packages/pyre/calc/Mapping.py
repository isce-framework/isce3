# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# class declaration
class Mapping:
    """
    Mix-in class that forms the basis of the representation of mappings

    Mappings are dictionaries with arbitrary keys whose values are nodes
    """


    # types
    from .exceptions import CircularReferenceError

    # constants
    category = "mapping"

    # public data
    @property
    def operands(self):
        """
        Iterate over my operands
        """
        # easy enough
        yield from self.data.values()
        # all done
        return


    # classifiers
    @property
    def mappings(self):
        """
        Return a sequence over mappings in my dependency graph
        """
        # i am one
        yield self
        # nothing further
        return


    # value management
    def getValue(self, **kwds):
        """
        Compute and return my value
        """
        # return the value of each operand
        return {name: op.value for name, op in self.data.items()}


    def setValue(self, value):
        """
        Add the {key, node} pair in {value} to the mapping
        """
        # unpack
        key, node = value
        # store
        self.data[key] = noe
        # all done
        return self


    # meta-methods
    def __init__(self, operands, **kwds):
        # chain up with an empty pile of dependencies
        super().__init__(operands=(), **kwds)
        # my operands are in a dict
        self.data = dict(**operands)
        # all done
        return


    def __getitem__(self, key):
        # return the value of the node stored under {key}
        return self.data[key].value


    def __setitem__(self, key, node):
        # store {node} under {key}
        self.data[key] = node
        # all done
        return


    # implementation details
    def _substitute(self, current, replacement):
        """
        Adjust the operands by substituting {replacement} for {current} in the set of operands
        """
        # go through my data
        for name, operand in self.data.items():
            # if we found the match
            if operand is current:
                # replace it
                self.data[name] = replacement

        # all done
        return self


# end of file
