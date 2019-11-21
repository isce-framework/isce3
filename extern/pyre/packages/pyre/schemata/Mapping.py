# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections.abc # for container identification
# superclass
from .Container import Container


# declaration
class Mapping(Container):
    """
    The base class for type declarators that map strings to other types
    """


    # constants
    typename = 'mapping' # the name of my type
    container = dict # the default container i represent
    complaint = 'could not coerce {0.value!r} to a mapping'


    # implementation details
    def _coerce(self, value, **kwds):
        """
        Convert {value} into a container
        """
        # string processing
        if isinstance(value, str):
            # otherwise, not supported
            raise NotImplementedError(
                "class {.__name__} cannot coerce strings".format(type(self)))

        # if we have a mapping
        if isinstance(value, collections.abc.Mapping):
            # go through each entry
            for key, entry in value.items():
                # convert it and hand it to the caller. perform the conversion incognito, in
                # case coercing my values requires the instantiation of components; i don't
                # want facilities to use the name of my node as the name of any instantiated
                # components
                yield key, self.schema.coerce(value=entry, incognito=True, **kwds)
            # all done
            return

        # if we have an iterable
        if isinstance(value, collections.abc.Iterable):
            # go through each entry
            for key, entry in value:
                # convert it and hand it to the caller. perform the conversion incognito, in
                # case coercing my values requires the instantiation of components; i don't
                # want facilities to use the name of my node as the name of any instantiated
                # components
                yield key, self.schema.coerce(value=entry, incognito=True, **kwds)
            # all done
            return

        # otherwise, flag it as bad input
        raise self.CastingError(value=value, description=self.complaint)


# end of file
