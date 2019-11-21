# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections.abc # for sequence checking
from .. import schemata # type information
# superclass
from .Slotted import Slotted


# declaration
@schemata.typed
class Property(Slotted):
    """
    The base class for traits that correspond to simple types
    """


    # framework data
    category = 'property'
    # predicate that indicates whether this trait is a property
    isProperty = True


    # mixins to be included to my type offering
    class schema:
        """Mixin for handling generic values"""

        # override the default expression handler
        @property
        def macro(self):
            """
            The default strategy for handling macros in slot values
            """
            # by default, build interpolations
            return self.pyre_nameserver.interpolation

        @property
        def native(self):
            """
            The strategy for building slots from more complex input values
            """
            return self.pyre_nameserver.variable


    class numeric:
        """Mixin for handling numeric types"""

        # override the default expression handler
        @property
        def macro(self):
            """
            Access to the default strategy for handling macros for numeric types
            """
            # build expressions
            return self.pyre_nameserver.expression


    class sequences:
        """Mixin for handling typed containers"""

        # override the default expression handler
        @property
        def macro(self):
            """
            The default strategy for handling slot values that are strings and therefore
            subject to some kind of evaluation in the context of the configuration store
            """
            # whatever my schema says
            return self.schema.macro

        def native(self, value, **kwds):
            """
            The strategy for building slots from more complex input values
            """
            # if the value is a sequence
            if isinstance(value, collections.abc.Iterable):
                # convert the items into nodes
                nodes = (self.schema.macro(value=item) for item in value)
                # and attach them to a sequence node
                return self.pyre_nameserver.sequence(nodes=nodes, **kwds)

            # if the value is {None}
            if value is None:
                # chain up
                return super().native(value=value, **kwds)

            # shouldn't get here
            assert False, 'unreachable'


    # meta-methods
    def __init__(self, classSlot=None, instanceSlot=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # build my slot factories
        self.classSlot = classSlot or self.factory(trait=self, processor=self.process)
        self.instanceSlot = instanceSlot or self.factory(trait=self, processor=self.process)
        # all done
        return


    def __str__(self):
        return "{0.name!r}: a property of type {0.typename!r}".format(self)


# end of file
