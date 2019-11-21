# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclasses
from .Communicator import Communicator


# declaration
class Cartesian(Communicator):
    """
    An encapsulation of Cartesian communicators
    """


    # per-instance public data
    axes = None
    periods = None
    coordinates = None


    # meta methods
    def __init__(self, capsule, axes, periods, reorder, **kwds):
        # build the capsule of the Cartesian communicator
        cartesian = self.mpi.communicatorCreateCartesian(capsule, reorder, axes, periods)

        # chain to my ancestors
        super().__init__(capsule=cartesian, **kwds)

        # save the rest
        self.axes = axes
        self.periods = periods
        # get my coördinates
        self.coordinates = self.mpi.communicatorCartesianCoordinates(
            self.capsule, self.rank, len(axes))

        # all done
        return


# end of file
