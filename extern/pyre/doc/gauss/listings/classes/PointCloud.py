# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class PointCloud:
    """
    The abstract base class for point generators
    """

    # interface
    def point(self, box):
        """
        Generate a random point on the interior of {box}

        parameters:
            {box}: pairs of points that specify the computational domain
        """
        raise NotImplementedError(
            "class {.__name__!r} should implement 'point'".format(type(self)))


# end of file
