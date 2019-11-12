# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


from .AbstractMetaclass import AbstractMetaclass


class Singleton(AbstractMetaclass):
    """
    Provide support for classes that create and manage a single instance

    Adapted from Michele Simionato's implementation
    """


    def __call__(cls, **kwds):
        """
        Build and return the singleton instance every time the constructor is called
        """
        # attempt to access the class instance
        try:
            return cls._pyre_singletonInstance
        except AttributeError:
            cls._pyre_singletonInstance = it = super().__call__(**kwds)
            return it


# end of file
