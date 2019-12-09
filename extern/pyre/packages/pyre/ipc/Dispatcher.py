# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre


# declaration
class Dispatcher(pyre.protocol, family="pyre.ipc.dispatchers"):
    """
    Protocol definition for components that monitor communication channels and invoke handlers
    when activity is detected
    """


    # default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The suggested implementation of the {Dispatcher} protocol
        """
        # {Selector} is the only choice currently
        from .Selector import Selector
        return Selector


    # interface
    @pyre.provides
    def watch(self):
        """
        Enter an indefinite loop of monitoring all registered event sources and invoking the
        registered event handlers
        """

    @pyre.provides
    def stop(self):
        """
        Stop monitoring all communication channels
        """

    # event scheduling
    @pyre.provides
    def alarm(self, interval, call):
        """
        Schedule {call} to be invoked after {interval} elapses. {interval} is expected to be
        a dimensional quantity from {pyre.units} with units of time
        """

    @pyre.export
    def whenReadReady(self, channel, call):
        """
        Add {call} to the list of routines to call when {channel} is ready to be read
        """

    @pyre.export
    def whenWriteReady(self, channel, call):
        """
        Add {call} to the list of routines to call when {channel} is ready to be written
        """

    @pyre.export
    def whenException(self, channel, call):
        """
        Add {call} to the list of routines to call when something exceptional has happened
        to {channel}
        """


# end of file
