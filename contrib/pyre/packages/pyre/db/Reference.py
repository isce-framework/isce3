# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from .Measure import Measure


# declaration
class Reference(Measure):
    """
    Representation of foreign keys
    """


    @property
    def decl(self):
        """
        SQL rendering of my type name
        """
        # my referent knows
        return self.referent.decl


    # interface
    def sql(self, value):
        """
        SQL rendering of {value}
        """
        # my referent knows
        return self.referent.sql(value=value)


    # markers
    def onDelete(self, action):
        """
        Set the action to perform when the target record is deleted. See {pyre.db.actions} for
        details
        """
        # mark
        self._foreign.delete = action
        # and return
        return

    def onUpdate(self, action):
        """
        Set the action to perform when the target record is updated. See {pyre.db.actions} for
        details
        """
        # mark
        self._foreign.update = action
        # and return
        return


    # meta-methods
    def __init__(self, **kwds):
        super().__init__()

        # set up my foreign key
        self._foreign = self.foreign(**kwds)

        # get the field reference recorded by the foreign key
        ref = self._foreign.reference
        # if the reference mentions a field explicitly
        if ref.field is not None:
            # save it
            field = ref.field
        # otherwise
        else:
            raise NotImplementedError("NYI!")

        # store my referent
        self.referent = field

        # all done
        return


# end of file
