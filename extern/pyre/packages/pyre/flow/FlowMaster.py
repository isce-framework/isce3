# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# framework
import pyre
# local support
from .NameGenerator import NameGenerator


# declaration
class FlowMaster(pyre.actor):
    """
    The meta-class of flow nodes
    """


    # public data
    pyre_nameGenerator = NameGenerator()


    # meta-methods
    def __call__(self, name=None, locator=None, implicit=False, **kwds):
        """
        Build an instance of one of my classes
        """
        # if this instance was not named explicitly by the user
        if implicit or name is None:
            # build a unique name
            name = self.pyre_nameGenerator.uid()

        # if necessary, build a locator to reflect the construction site; the one {Actor} would
        # build would be wrong since it is now one level deeper in the stack
        locator = pyre.tracking.here(1) if locator is None else locator

        # create the instance
        instance = super().__call__(name=name, locator=locator, implicit=implicit, **kwds)
        # and return it
        return instance


# end of file
