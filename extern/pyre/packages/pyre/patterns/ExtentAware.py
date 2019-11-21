# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to weak references
import weakref
# my superclass
from .AbstractMetaclass import AbstractMetaclass


class ExtentAware(AbstractMetaclass):
    """
    Metaclass that endows its instances with awareness of their extent.

    The extent of a class is the set of its own instances and the instances of all of its
    subclasses.

    The class extent is stored with the first class that mentions {ExtentAware} as a metaclass,
    and all descendants are counted by that class. Descendants that want to keep track of their
    own extent and prevent their extent from being counted by their superclass must be declared
    with {pyre_extentRoot} set to {True}.

    implementation details:

        __new__: intercept the creation of the client class record and add the reference
        counting weak set as a class attribute

        __call__: capture the creation of an instance, since it is this method that triggers
        the call to the client class' constructor. we let super().__call__ build the instance
        and then add a weak reference to it in _pyre_extent
    """


    # class methods
    def __init__(self, name, bases, attributes, pyre_extentRoot=False, **kwds):
        """
        Endow extent aware class records with a registry of their instances

        By default, an extent aware class keeps track of both its own instances and the
        instances of all of its subclasses. Descendants that wish to maintain their own count
        """
        # chain up
        super().__init__(name, bases, attributes, **kwds)

        # add the weakset attribute that maintains the extent, if it is not already there; this
        # has the effect of storing the class extent at the root class in a hierarchy which
        # makes it easy to check that descendants have been garbage collected as well. if you
        # want to keep track of the extent at some other point in a class hierarchy, declare
        # that class with {pyre_extentRoot} set to {True}
        if pyre_extentRoot or not hasattr(self, "_pyre_extent"):
            # build the instance registry
            self._pyre_extent = weakref.WeakSet()
        # all done
        return


    def __call__(self, *args, **kwds):
        """
        Intercept the call to the client constructor, build the instance and keep a weak
        reference to it
        """
        # build the instance
        instance = super().__call__(*args, **kwds)
        # add it to the class extent
        self._pyre_extent.add(instance)
        # and return it
        return instance


# end of file
