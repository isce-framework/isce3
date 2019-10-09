# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections


# declaration
class Foundry:
    """
    A decorator for callables that return component classes
    """


    # public data
    pyre_tip = '' # a short description of what this component does
    pyre_factory = None # the actual callable that returns the component class
    pyre_implements = None # the protocols implemented by this component


    # meta-methods
    def __new__(cls, factory=None, implements=None, tip='', **kwds):
        """
        Trap the invocation with meta-data and delay the decoration of the callable
        """
        # if the method is known
        if factory is not None:
            # check that the user gave us something we can decorate
            assert callable(factory), 'please invoke with keyword arguments'
            # and chain up to do the normal thing; swallow the extra arguments, but don't
            # worry, we'll see them again in {__init__}
            return super().__new__(cls, **kwds)

        # if we don't know the factory, we were invoked with keyword arguments; the strategy
        # here is to return a {Foundry} constructor as the value of this invocation, which
        # accomplishes two things: it gives python something to call when the method
        # declaration is done, and prevents my {__init__} from getting invoked prematurely

        # here is the constructor closure
        def build(factory):
            """
            Convert a component factory into a foundry
            """
            # just build one of my instances
            return cls(factory=factory, implements=implements, tip=tip, **kwds)

        # to hand over
        return build


    def __init__(self, factory, implements=None, tip='', **kwds):
        # chain up
        super().__init__(**kwds)
        # appropriate the factory's docstring
        self.__doc__ = factory.__doc__
        # save the original callable
        self.pyre_factory = factory
        # the tip
        self.pyre_tip = tip or self.__doc__
        # and the list of implemented protocols
        self.pyre_implements = self.sequify(implements)
        # all done
        return


    def __call__(self, *args, **kwds):
        # invoke the factory and return the result
        return self.pyre_factory(*args, **kwds)


    # implementation details
    @classmethod
    def sequify(self, items):
        """
        Normalize {items} into a tuple
        """
        # take care of {None} and anything false or empty
        if not items: return ()
        # if {items} is any iterable
        if isinstance(items, collections.abc.Iterable):
            # turn it into a tuple
            return tuple(items)
        # otherwise, place the lone item into a tuple
        return (items,)


# end of file
