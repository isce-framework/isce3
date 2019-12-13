# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# my superclass
from ..framework.Dashboard import Dashboard # access to the framework managers


# class declaration
class SlotFactory(Dashboard):
    """
    A factory of slots of a given trait
    """


    # get the identity value processor
    from ..schemata import identity
    # make a null value processor
    noop = identity().coerce


    # meta-methods
    def __init__(self, trait, pre=noop, post=noop, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my parts
        self.trait = trait
        self.pre = pre
        self.post = post
        # all done
        return


    def __call__(self, value, current=None, **kwds):
        """
        Make a slot for my client trait
        """
        # if the {value} is already a slot
        if isinstance(value, self.pyre_nameserver.node):
            # just use it
            new = value
        # if it is a string
        elif isinstance(value, str):
            # do whatever the trait specifies as the slot building factory for string input
            new = self.trait.macro(preprocessor=self.pre, postprocessor=self.post,
                                   value=value, **kwds)
        # anything else
        else:
            # is native to the trait
            new = self.trait.native(preprocessor=self.pre, postprocessor=self.post,
                                    value=value, **kwds)

        # if the existing slot is non trivial
        if current is not None:
            # replace it
            new.replace(obsolete=current)

        # all done
        return new


# end of file
