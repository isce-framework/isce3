# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre


# class declaration
class Specification(pyre.protocol):
    """
    The protocol of product specifications

    Specifications are snapshots of product attributes
    """


    # public data
    input = False
    output = False


    # facility makers
    @classmethod
    def blueprint(cls, **kwds):
        """
        Make an input descriptor
        """
        # ask my meta-class to build a descriptor
        facility = cls(**kwds)
        # mark it as input
        facility.input = True
        # and return it
        return facility


    @classmethod
    def product(cls, **kwds):
        """
        Make an output descriptor
        """
        # ask my meta-class to build a descriptor
        facility = cls(**kwds)
        # mark it as input
        facility.output = True
        # and return it
        return facility


# end of file
