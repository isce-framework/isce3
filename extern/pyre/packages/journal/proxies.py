# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the C++ extension; guaranteed to exist since otherwise this modules would not
# be in the process of being imported
from . import journal

# factories

# these index factories bring together two pieces of information that are required for the
# extension module to access {libjournal} correctly: the severity of the channel and its
# inventory type.
def debugIndex():
    """
    Build an object that is a wrapper around the debug channel index from the C++
    extension. i.e. {pyre::journal::debug_t::index_t}
    """
    #
    return Index(lookup=journal.lookupDebugInventory, inventory=Disabled)


def firewallIndex():
    """
    Build an object that is a wrapper around the firewall channel index from the C++
    extension. i.e. {pyre::journal::firewall_t::index_t}
    """
    #
    return Index(lookup=journal.lookupFirewallInventory, inventory=Enabled)


def infoIndex():
    """
    Build an object that is a wrapper around the info channel index from the C++
    extension. i.e. {pyre::journal::info_t::index_t}
    """
    #
    return Index(lookup=journal.lookupInfoInventory, inventory=Enabled)


def warningIndex():
    """
    Build an object that is a wrapper around the warning channel index from the C++
    extension. i.e. {pyre::journal::warning_t::index_t}
    """
    #
    return Index(lookup=journal.lookupWarningInventory, inventory=Enabled)


def errorIndex():
    """
    Build an object that is a wrapper around the error channel index from the C++
    extension. i.e. {pyre::journal::error_t::index_t}
    """
    #
    return Index(lookup=journal.lookupErrorInventory, inventory=Enabled)


# implementation details
class Index:
    """
    Wrapper around the C++ diagnostic indices from the journal extension module
    """

    # meta methods
    def __init__(self, lookup, inventory):
        # keep it simple...
        self.lookup = lookup
        self.inventory = inventory
        # all done
        return

    # behave like a dictionary
    def __getitem__(self, name):
        """
        Retrieve the state associated with the given {name}
        """
        return self.inventory(capsule=self.lookup(name))


class Enabled:
    """Wrapper around {pyre::journal::Inventory<true>}"""

    # public state
    device = None

    @property
    def state(self):
        return journal.getEnabledState(self.capsule)

    @state.setter
    def state(self, value):
        return journal.setEnabledState(self.capsule, value)


    def __init__(self, capsule):
        self.capsule = capsule
        return

class Disabled:
    """Wrapper around {pyre::journal::Inventory<false>}"""

    # public state
    device = None

    @property
    def state(self):
        return journal.getDisabledState(self.capsule)

    @state.setter
    def state(self, value):
        return journal.setDisabledState(self.capsule, value)


    def __init__(self, capsule):
        self.capsule = capsule
        return


# end of file
