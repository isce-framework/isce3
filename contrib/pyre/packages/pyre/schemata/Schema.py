# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# declaration
class Schema:
    """
    The base class for type declarators
    """


    # exception
    from .exceptions import CastingError

    # constants
    typename = 'identity'


    # public data
    @property
    def default(self):
        """
        Grant access to my default value
        """
        # easy enough
        return self._default

    @default.setter
    def default(self, value):
        """
        Save {value} as my default
        """
        # also easy
        self._default = value
        # all done
        return


    # interface
    def coerce(self, value, **kwds):
        """
        Convert the given value into a python native object
        """
        # just leave it alone
        return value


    def string(self, value):
        """
        Render value as a string that can be persisted for later coercion
        """
        # respect {None}
        if value is None: return None
        # my value knows
        return str(value)


    def json(self, value):
        """
        Generate a JSON representation of {value}
        """
        # respect {None}
        if value is None: return None
        # by default, let the raw value through; the schemata that are not JSON representable
        # must override to provide something suitable
        return value


    # meta-methods
    def __init__(self, default=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my default value
        self._default = default
        # all done
        return


# end of file
