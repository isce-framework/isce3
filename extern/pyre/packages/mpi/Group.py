# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# base class
from .Object import Object
# meta-class
from pyre.patterns.ExtentAware import ExtentAware


# declaration
class Group(Object, metaclass=ExtentAware):
    """
    Encapsulation of MPI communicator groups
    """

    # class level public data
    undefined = Object.mpi.undefined


    # per-instance public data
    rank = 0 # my rank in this group
    size = 0 # the size of this group


    # check whether a group is empty
    def isEmpty(self):
        """
        Check whether i am an empty group
        """
        return self.mpi.groupIsEmpty(self.capsule)


    # building groups using explicit ranklists
    def include(self, included):
        """
        Build a group out of the processes in {included}
        """
        # build a new group capsule
        capsule = self.mpi.groupInclude(self.capsule, tuple(included))
        # check whether it is a valid group
        if capsule:
            # wrap it and return it
            return Group(capsule=capsule)
        # otherwise return an invalid group
        return None


    def exclude(self, excluded):
        """
        Build a group out of all processes except those in {excluded}
        """
        # build a new group capsule
        capsule = self.mpi.groupExclude(self.capsule, tuple(excluded))
        # check whether it is a valid group
        if capsule:
            # wrap it and return it
            return Group(capsule=capsule)
        # otherwise return an invalid group
        return None


    # the set-like operations
    def union(self, g):
        """
        Build a new group whose processes are the union of mine and {g}'s
        """
        # build the new group capsule
        capsule = self.mpi.groupUnion(self.capsule, g.capsule)
        # check whether it is a valid group
        if capsule:
            # wrap it and return it
            return Group(capsule=capsule)
        # otherwise
        return None


    def intersection(self, g):
        """
        Build a new group whose processes are the intersection of mine and {g}'s
        """
        # build the new group capsule
        capsule = self.mpi.groupIntersection(self.capsule, g.capsule)
        # check whether it is a valid group
        if capsule:
            # wrap it and return it
            return Group(capsule=capsule)
        # otherwise
        return None


    def difference(self, g):
        """
        Build a new group whose processes are the difference of mine and {g}'s
        """
        # build the new group capsule
        capsule = self.mpi.groupDifference(self.capsule, g.capsule)
        # check whether it is a valid group
        if capsule:
            # wrap it and return it
            return Group(capsule=capsule)
        # otherwise
        return None


    # meta methods
    def __init__(self, capsule, **kwds):
        # chain to my ancestors
        super().__init__(**kwds)

        # store my attributes
        self.capsule = capsule
        # and precompute my rank and size
        self.rank = self.mpi.groupRank(capsule)
        self.size = self.mpi.groupSize(capsule)

        # all done
        return


    # implementation details
    capsule = None


# end of file
