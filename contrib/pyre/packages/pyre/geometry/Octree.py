# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# externals
import itertools


# class declaration
class Octree:
    """
    A dimension independent implementation of the octree family of spatial data structures
    """


    def contains(self, point):
        """
        Determine whether {point} falls within my box
        """
        # make sure that each coordinate
        for p, (low, high) in zip(point, self.intervals):
            # is within my bounds
            if p < low or p > high: return False
        # all done
        return True


    def insert(self, point, level=0):
        """
        Insert {point} within this box, creating any children necessary
        """
        # print("{}: inserting {} in {}".format(level, point, self.intervals))
        # if the point does not belong to me
        if not self.contains(point):
            # complain
            raise ValueError("point {} outside box {}".format(point, self.intervals))
        # if I am empty
        if self.point is None and self.branches is None:
            # store the point
            self.point = point
            # all done
            return level
        # if I have a point
        mine = level
        if self.point:
            # check that i don't have any children yet
            assert self.branches is None
            # build my branches
            self.branches = self.subdivide()
            # locate the correct branch for my point
            branch = self.branches[self.hash(self.point)]
            # insert my point
            mine = branch.insert(self.point, level+1)
            # clear it
            self.point = None
        # locate the correct branch for the new point
        branch = self.branches[self.hash(point)]
        # insert it
        new = branch.insert(point, level+1)
        # all done
        return max(mine, new)


    def subdivide(self):
        """
        Convert me from a leaf node to an internal one. This version build all children
        regardless of whether they end up containing points
        """
        # my subintervals
        subs = (((lo, (lo+hi)/2), ((lo+hi)/2, hi)) for lo,hi in self.intervals)
        # use the cartesian product to build my branches
        return tuple(octree(intervals=box) for box in itertools.product(*subs))


    def hash(self, point):
        """
        Figure out within which of my children {point} falls
        """
        # initialize the index
        index = 0
        # loop over my intervals
        for p, (lo,hi) in zip(point, self.intervals):
            # left shift the index
            index *= 2
            # compute the offset
            if p > (lo+hi)/2: index += 1
        # all done
        return index


    def __init__(self, intervals, **kwds):
        super().__init__(**kwds)
        self.point = None
        self.branches = None
        self.intervals = intervals
        # all done
        return


# end of file
