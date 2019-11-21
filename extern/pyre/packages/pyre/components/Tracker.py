# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import collections
# support
from .Revision import Revision
# superclass
from .Monitor import Monitor

# value tracker
class Tracker(Monitor):
    """
    A class that monitors the traits of a set of components and maintains a record of their values
    """


    # interface
    def track(self, component):
        """
        Add the {component} traits to the pile of observables
        """
        # watch the component's traits
        self.watch(component=component)

        # grab my history
        history = self.history
        # get the component inventory so that we can store the current value of its traits
        inventory = component.pyre_inventory
        # and its nameserver, so we can grab the value meta-data
        nameserver = component.pyre_nameserver
        # go through the slots that store the value of its traits
        for slot in inventory.getSlots():
            # grab the key
            key = slot.key
            # ask the name server for the slot meta-data
            info = nameserver.getInfo(key)
            # save the info
            revision = Revision(value=slot.value, locator=info.locator, priority=info.priority)
            # record
            history[key].append(revision)

        # all done
        return self


    # hooks
    def flush(self, observable, **kwds):
        """
        Handle the notification that the value of {observable} has changed
        """
        # get the slot key
        key = observable.key
        # ask it for its value
        value = observable.value
        # and the meta-data maintained by the nameserver
        info = observable.pyre_nameserver.getInfo(key)
        # save the info
        revision = Revision(value, locator=info.locator, priority=info.priority)
        # record
        self.history[key].append(revision)
        # chain up
        return super().flush(observable=observable, **kwds)


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # initialize my history
        self.history = collections.defaultdict(list)
        # all done
        return


# end of file
