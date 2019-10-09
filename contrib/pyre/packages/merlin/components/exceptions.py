# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# get the framework
import pyre


# the base class for all merlin exceptions
class MerlinError(pyre.PyreError):
    """
    Base class for merlin exceptions
    """

# derived ones
class SpellNotFoundError(MerlinError):
    """
    Exception raised when the requested spell cannot be located
    """

    # public data
    description = "spell {.spell!r} not found"

    # meta-methods
    def __init__(self, spell, **kwds):
        # chain up
        super().__init__(**kwds)
        # save the missing spell name
        self.spell = spell
        # all done
        return


# end of file
