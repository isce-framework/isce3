# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#



# class declaration
class Filter:
    """
    A mix-in class that changes the values of nodes iff they pass its constraints
    """


    # value management
    def setValue(self, **kwds):
        """
        Override the value setter to let the assignment go through iff my constraints are all
        satisfied
        """
        # go through all my constraints
        for constraint in self.constraints:
            # if any fail, leave the value unmodified
            if not constraint(node=self, **kwds): return self

        # if they all succeed, update the value
        return super().setValue(**kwds)


    # meta-methods
    def __init__(self, value=None, constraints=None, **kwds):
        # initialize my constraints; must be done first because {setValue} is invoked as part
        # of construction to process in the incoming value
        self.constraints = [] if constraints is None else constraints
        # chain up
        super().__init__(value=value, **kwds)
        # all done
        return


# end of file
