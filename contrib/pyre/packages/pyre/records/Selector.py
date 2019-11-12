# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class Selector:
    """
    The base class for objects responsible for providing named access to field descriptors
    """


    # public data
    index = None # the index of my value in the data tuple
    field = None # the associated descriptor with the meta data


    # meta-methods
    def __init__(self, index, field, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my spot
        self.index = index
        self.field = field
        # all done
        return


    def __get__(self, record, cls):
        """
        Field retrieval
        """
        # return my meta-data regardless of the target of this access
        return self.field


    def __set__(self, record, value):
        """
        Field modification
        """
        # complain
        raise NotImplementedError('field selectors do not support write access')


# end of file
