# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


class AbstractMetaclass(type):
    """
    The base metaclass from which all pyre metaclasses derive.

    The main raison d'être for this class is to lift the constraint that the signatures of the
    various metaclass hooks must absorb all arguments passed through {**kwds} before invoking
    their implementations in {type}.

    implementation details:
      __new__: swallow the **kwds that {type} does not recognize
      __init__: swallow the **kwds that {type} does not recognize
    """


    # meta methods
    def __new__(cls, name, bases, attributes, **kwds):
        """
        Swallow **kwds and call type.__new__
        """
        return super().__new__(cls, name, bases, attributes)


    def __init__(self, name, bases, attributes, **kwds):
        """
        Swallow **kwds and call type.__init__
        """
        return super().__init__(name, bases, attributes)


# end of file
