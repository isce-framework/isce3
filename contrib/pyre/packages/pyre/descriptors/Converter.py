# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Processor import Processor


# declaration
class Converter(Processor):
    """
    A record method decorator that registers this method as a converter of descriptor values
    """


    # meta-methods
    def __call__(self, method):
        """
        Add {method} as a converter to my registered descriptors
        """
        # go through the sequence of registered descriptors
        for trait in self.traits:
            # and register {method} as a converter
            trait.converters.append(method)
        # all done
        return method


# end of file
