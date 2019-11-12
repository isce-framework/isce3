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
    def points(self, n, box):
        """
        Generate {n} random points on the interior of {box}

        parameters:
            {n}: the number of points to generate
            {box}: a pair of points that specify the computational domain
        """
        raise NotImplementedError(
            "class {.__name__!r} should implement 'points'".format(type(self)))


# end of file
