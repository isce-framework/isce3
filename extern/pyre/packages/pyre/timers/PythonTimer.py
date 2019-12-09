# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import time


# ancestor
from .AbstractTimer import AbstractTimer


# declaration
class PythonTimer(AbstractTimer):
    """
    A simple timer implementation based on the facilities from the time module in the python
    standard library
    """


    # interface
    def start(self):
        """
        Start the timer
        """
        # take a reading from the system clock
        self._start = time.time()
        # enable chaining
        return self


    def stop(self):
        """
        Stop the timer
        """
        # update the accumulated time
        self._accumulatedTime += time.time() - self._start
        # disable the timer
        self._start = None
        # and return
        return self


    def reset(self):
        """
        Reset the timer.

        This sets the time accumulated by this timer to zero and disables it
        """
        # zero out the accumulated time
        self._accumulatedTime = 0
        # disable the timer
        self._start = None
        # and return
        return self


    def read(self):
        """
        Return the total time this timer has been running
        """
        # return the accumulated time
        return self._accumulatedTime


    def lap(self):
        """
        Read the total time this timer has been running without disturbing it
        """
        # take a reading from the system clock
        now = time.time()
        # and return the total time accumulated so far
        return self._accumulatedTime + (now - self._start)


    # meta methods
    def __init__(self, **kwds):
        # chain to the ancestors
        super().__init__(**kwds)

        # data
        # mark as uninitialized; this disables arithmetic with dead timers
        self._start = None
        self._accumulatedTime = 0

        # all done
        return


    # private data
    _start = 0
    _accumulatedTime = 0


# end of file
