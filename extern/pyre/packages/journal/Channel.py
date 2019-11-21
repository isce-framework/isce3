# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# super-classes
from pyre.patterns.Named import Named


# declaration
class Channel(Named):
    """
    This class encapsulates access to the shared channel state
    """


    # class data
    # the anchor component with the configurable parts
    journal = None # patched by the journal bootstrapping process


    # public data
    @property
    def active(self):
        """
        Get my current activation state
        """
        return self._inventory.state


    @active.setter
    def active(self, state):
        """
        Set my current activation state
        """
        # save the new state
        self._inventory.state = state
        # and return
        return


    @property
    def device(self):
        """
        Get my current output device
        """
        # first, check the specific device assigned to my channel
        device = self._inventory.device
        # if one was assigned, return it
        if device is not None: return device
        # otherwise, issue a request for the default device
        return self.journal.device


    @device.setter
    def device(self, device):
        """
        Associate a device to be used for my output
        """
        # attach the new device to my shared state
        self._inventory.device = device
        # and return
        return


    # interface
    def activate(self):
        """
        Mark me as active
        """
        # do it
        self.active = True
        # and return
        return self


    def deactivate(self):
        """
        Mark me as inactive
        """
        # do it
        self.active = False
        # and return
        return self


    # meta methods
    def __init__(self, name, **kwds):
        # chain to my ancestors
        super().__init__(name=name, **kwds)
        # look up my shared state
        self._inventory = self._index[name]
        # and return
        return


    def __bool__(self):
        """
        Simplify the state testing
        """
        # delegate to my state getter
        return self.active


    # implementation details
    # private class data
    _index = None

    # subclasses must supply a non-trivial implementation of the mechanism by which state
    # shared among channel instances is managed. when the C++ extension is not available at
    # runtime, this package defaults to a pure python implementation that uses a {defaultdict};
    # this implies that we need access to factories that build instances with the correct
    # default activation state, hence the two nested class declarations below. their names
    # reflect their default state, in the absence of any configuration instructions by the
    # user. when the state is False, the channel does not produce any output; when device is
    # {None}, the default device is used instead
    class Enabled:
        """Shared state for channels that are enabled by default"""
        state = True
        device = None

    class Disabled:
        """Shared state for channels that are disabled by default"""
        state = False
        device = None


# end of file
