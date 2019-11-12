# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the package
import merlin


# a spell
class sample(merlin.spell, family="merlin.spells.sample"):
    """
    A sample spell
    """
    @merlin.export
    def main(self):
        return "{.pyre_name}: main".format(self)


# end of file
