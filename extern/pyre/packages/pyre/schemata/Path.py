# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
from .. import primitives
# superclass
from .Schema import Schema


# my declaration
class Path(Schema):
    """
    A type declarator for paths
    """


    # constants
    typename = 'path' # the name of my type
    complaint = "cannot cast {0.value!r} into a path"

    # magic values
    cwd = primitives.path.cwd
    root = primitives.path.root
    home = primitives.path.home()


    # interface
    def coerce(self, value, **kwds):
        """
        Attempt to convert {value} into a path
        """
        # perhaps it is already a path
        if isinstance(value, primitives.path):
            # in which case, just leave it alone
            return value

        # the rest assume {value} is a string; if it isn't
        if not isinstance(value, str):
            # complain
            raise self.CastingError(value=value, description=self.complaint)

        # cast {value} into a path
        return primitives.path(value)


    def json(self, value):
        """
        Generate a JSON representation of {value}
        """
        # represent as a string
        return self.string(value)


# end of file
