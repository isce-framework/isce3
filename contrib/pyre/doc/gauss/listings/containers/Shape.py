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
    def interior(self, points):
        """
        Discard {points} that are on my exterior
        """
        raise NotImplementedError(
            "class {.__name__!r} should implement 'interior'".format(type(self)))


# end of file
