# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the extension
from ..extensions import timers


# superclasses
from .AbstractTimer import AbstractTimer


# declaration
class NativeTimer(AbstractTimer):
    """
    Timer implementation that relies on the timer extension module
    """


    # interface
    def start(self):
        """
        Start the timer
        """
        # take a reading from the system clock
        timers.start(self._timer)
        # enable chaining
        return self


    def stop(self):
        """
        Stop the timer
        """
        # stop the timer
        timers.stop(self._timer)
        # and return
        return self


    def reset(self):
        """
        Reset the timer.

        This sets the time accumulated by this timer to zero and disables it
        """
        # stop the timer
        timers.reset(self._timer)
        # and return
        return self


    def read(self):
        """
        Return the total time this timer has been running
        """
        # return the accumulated time
        return timers.read(self._timer)


    def lap(self):
        """
        Read the total time this timer has been running without disturbing it
        """
        # return the total time accumulated so far
        return timers.lap(self._timer)


    # meta methods
    def __init__(self, name, **kwds):
        # chain to the ancestors
        super().__init__(name=name, **kwds)

        # create a new timer capsule
        self._timer = timers.newTimer(name)

        # all done
        return


    # implementation details
    _timer = None

# end of file
