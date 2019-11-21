# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Trait import Trait


# declaration
class Behavior(Trait):
    """
    The base class for component methods that are part of its external interface
    """


    # public data
    method = None # the actual callable in the component declaration


    # framework data
    # my category name
    category = 'behavior'
    # predicate that indicates whether this trait is a behavior
    isBehavior = True


    # meta-methods
    def __new__(cls, method=None, tip=None, **kwds):
        """
        Trap the invocation with meta-data and delay the decoration of the method
        """
        # if the method is known
        if method is not None:
            # check that the user gave us something we can decorate
            assert callable(method), 'please invoke with keyword arguments'
            # and chain up to do the normal thing; swallow the extra arguments, but don't
            # worry, we'll see them again in {__init__}
            return super().__new__(cls, **kwds)

        # if we don't know the method, we were invoked with keyword arguments; the strategy
        # here is to return a {Behavior} constructor as the value of this invocation, which
        # accomplishes two things: it gives python something to call when the method
        # declaration is done, and prevents my {__init__} from getting invoked prematurely

        # here is the constructor closure
        def build(method):
            """
            Convert a component method into a behavior
            """
            # just build one of my instance
            return cls(method=method, tip=tip, **kwds)

        # to hand over
        return build


    def __init__(self, method, tip=None, **kwds):
        # chain up
        super().__init__(**kwds)
        # appropriate the method's docstring
        self.__doc__ = method.__doc__
        # save it
        self.method = method
        # save the tip
        self.tip = tip
        # all done
        return


    def __get__(self, instance, cls):
        """
        Access to the behavior
        """
        # dispatch to the encapsulated method
        return self.method.__get__(instance, cls)


    def __set__(self, instance, value):
        """
        Disable writing to behavior descriptors
        """
        raise TypeError(
            "can't modify {.name!r}, part of the public interface of {.pyre_name!r}"
            .format(self, instance))


    def __str__(self):
        return "{0.name!r}: a behavior".format(self)


# end of file
