# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the framework
import pyre

# declaration
class PointCloud(pyre.protocol, family="gauss.meshes"):
    """
    Protocol declarator for an unstructured collection of points
    """

    # my default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default {PointCloud} implementation
        """
        # use the built in random number generator
        from .Mersenne import Mersenne
        return Mersenne

    # interface
    @pyre.provides
    def points(self, count, box):
        """
        Generate {count} random points on the interior of {box}
        """


# end of file
