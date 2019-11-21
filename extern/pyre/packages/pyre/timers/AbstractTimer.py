# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# ancestor
from ..patterns.Named import Named


# declaration
class AbstractTimer(Named):
    """
    Base class for timers
    """


    # interface
    def start(self):
        """
        Start the timer
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'start'".format(type(self)))


    def stop(self):
        """
        Stop the timer
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'stop'".format(type(self)))


    def reset(self):
        """
        Reset the timer. Resetting a time disables it and clears the accumulated time. The timer
        must be started again before it can read
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'reset'".format(type(self)))


    def read(self):
        """
        Read the time elapsed between a start and a stop, in seconds. The accuracy of the timer is
        determined by the implementation strategy
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'read'".format(type(self)))


    def lap(self):
        """
        Read the time accumulated by this timer without disturbing it
        """
        raise NotImplementedError(
            "class {.__name__!r} must implement 'lap'".format(type(self)))


# end of file
