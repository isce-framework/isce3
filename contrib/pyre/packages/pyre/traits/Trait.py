# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclasses
from ..descriptors import stem # for types and algebra
from ..framework.Dashboard import Dashboard # access to the framework managers


# declaration
class Trait(stem.variable, Dashboard):
    """
    This is the base class for component features that form their public interface

    Traits extend the notion of a class attribute to an object that is capable of
    capturing information about the attribute that has no natural resting place as part of a
    normal class declaration.

    Traits enable end-user configurable state, for both simple attributes and references to
    more elaborate objects, such as other components. Individual inventory items need a name
    that enables access to the associated information, per-instance actual storage for the
    attribute value, and additional meta data, such as reasonable default values when the
    attribute is not explicitly given a value during configuration, and the set of constraints
    it should satisfy before it is considered a legal value.
    """

    # N.B.: resist the temptation to add abstract definitions for {__get__} and {__set__} in
    # this base class; their mere presence changes the semantics and gets in the way of the
    # trait name translation that is necessary to support trait aliases

    # framework data
    # my category
    category = 'trait' # the stem cell of traits...
    # predicate that indicates whether this trait is a behavior
    isBehavior = False
    # predicate that indicates whether this trait is a property
    isProperty = False
    # predicate that indicates whether this trait is a facility
    isFacility = False
    # predicate that indicates whether this trait is subject to runtime configuration
    isConfigurable = False


    # framework support
    def classConfigured(self, **kwds):
        """
        Notification that the component class I am being attached to is configured
        """
        # nothing to do, by default
        return self


    def instanceConfigured(self, **kwds):
        """
        Notification that the component instance I am being attached to is configured
        """
        # nothing to do, by default
        return self


# end of file
