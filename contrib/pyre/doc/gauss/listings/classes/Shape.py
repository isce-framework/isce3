# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Shape:
    """
    The abstract base class for representations of geometrical regions
    """

    # interface
    def interior(self, point):
        """
        Predicate that checks whether {point} falls on my interior
        """
        raise NotImplementedError(
            "class {.__name__!r} should implement 'interior'".format(type(self)))


# end of file
