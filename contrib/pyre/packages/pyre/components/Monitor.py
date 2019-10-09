# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# superclass
from pyre.calc.Probe import Probe


# value monitor
class Monitor(Probe):
    """
    A probe that monitors the traits of a set of components
    """


    # interface
    def watch(self, component):
        """
        Monitor {component} for changes in the values of its traits
        """
        # add its trait slots to my observables
        return self.observe(observables=component.pyre_inventory.getSlots())


# end of file
