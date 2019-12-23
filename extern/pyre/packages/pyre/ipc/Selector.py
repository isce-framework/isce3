# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import select
import collections
# my interface
from . import dispatcher
# my base class
from .Scheduler import Scheduler


# declaration
class Selector(Scheduler, family='pyre.ipc.dispatchers.selector', implements=dispatcher):
    """
    An event demultiplexer implemented using the {select} system call.

    In addition to supporting alarms via its {Scheduler} base class, {Selector} monitors
    changes in the state of channels. Processes that hold {Selector} instances can go to sleep
    until either an alarm rings or a channel is ready for IO, at which point {Selector} invokes
    whatever handler is associated with the event.
    """


    # interface
    @pyre.export
    def whenReadReady(self, channel, call):
        """
        Add {call} to the list of routines to call when {channel} is ready to be read
        """
        # add it to the pile
        self._read[channel.inbound].append(self._event(channel=channel, handler=call))
        # and return
        return


    @pyre.export
    def whenWriteReady(self, channel, call):
        """
        Add {call} to the list of routines to call when {channel} is ready to be written
        """
        # add it to the pile
        self._write[channel.outbound].append(self._event(channel=channel, handler=call))
        # and return
        return


    @pyre.export
    def whenException(self, channel, call):
        """
        Add {call} to the list of routines to call when something exceptional has happened
        to {channel}
        """
        # add both endpoints to the pile
        self._exception[channel.inbound].append(self._event(channel=channel, handler=call))
        self._exception[channel.outbound].append(self._event(channel=channel, handler=call))
        # and return
        return


    @pyre.export
    def stop(self):
        """
        Request the selector to stop watching for further events
        """
        # adjust my state
        self._watching = False
        # and return
        return


    @pyre.export
    def watch(self):
        """
        Enter an indefinite loop of monitoring all registered event sources and invoking the
        registered event handlers
        """
        # reset my state
        self._watching = True
        # grab a channel
        channel = self._debug
        # until someone says otherwise
        while self._watching:
            # show me
            channel.line("watching:")
            # compute how long i am allowed to be asleep
            channel.line(f"    computing the allowed sleep interval")
            timeout = self.poll()
            channel.line(f"    max sleep: {timeout}")

            # construct the descriptor containers
            channel.line("    collecting the event sources")
            iwtd = self._read.keys()
            owtd = self._write.keys()
            ewtd = self._exception.keys()

            # if my channel is active
            if channel:
                # show me the descriptors that have data to read
                if iwtd: channel.line(f"      read:")
                for fd in iwtd:
                    for event in self._read[fd]:
                        channel.line(f"        {event.channel}")
                # show me the channels that are ready to be written
                if owtd: channel.line("      write:")
                for fd in owtd:
                    for event in self._write[fd]:
                        channel.line(f"        {event.channel}")
                # show me the channels with exceptions
                if ewtd: channel.line("      exception:")
                for channel in ewtd:
                    for event in self._exception[fd]:
                        channel.line(f"        {event.channel}")

            # check for indefinite block
            channel.line("    checking for indefinite block")
            if not iwtd and not owtd and not ewtd and timeout is None:
                channel.log("** no registered handlers left; exiting")
                return

            # show me
            channel.log(f"    calling select; timeout={timeout}")
            # wait for an event
            try:
                reads, writes, excepts = select.select(iwtd, owtd, ewtd, timeout)
            # when a signal is delivered to a handler registered by the application, the select
            # call is interrupted and raises {InterruptedError}, a subclass of {OSError}
            except InterruptedError as error:
                # unpack
                errno = error.errno
                msg = error.strerror
                # show me
                channel.line(f"signal received: errno={errno}: {msg}")
                channel.line(f"  more watching: {self._watching}")
                channel.log()
                # keep going
                continue

            # if my channel is active
            if channel:
                # show me
                channel.line("activity detected:")
                # some details
                channel.line(f"      read clients: {len(reads)}")
                channel.line(f"      write clients: {len(writes)}")
                channel.line(f"      except clients: {len(excepts)}")
                # flush
                channel.log()

            # dispatch to the handlers of file events
            channel.log("    dispatching to handlers")
            self.dispatch(index=self._exception, entities=excepts)
            self.dispatch(index=self._write, entities=writes)
            self.dispatch(index=self._read, entities=reads)

            # raise the overdue alarms
            channel.log(f"    raising alarms: {len(self._alarms)} registered")
            self.awaken()

            # flush
            channel.log("moving on...")

        # sign off
        channel.log("done watching")
        # all done
        return


    def dispatch(self, index, entities):
        """
        Invoke the handlers registered in {index} that are associated with the descriptors in
        {entities}
        """
        # iterate over the active entities
        for active in entities:
            # invoke the event handlers and save the events whose handlers return {True}
            events = list(
                event for event in index[active]
                if event.handler(channel=event.channel)
                )
            # if no handlers requested to be rescheduled
            if not events:
                # remove the descriptor from the index
                del index[active]
            # otherwise
            else:
                # reschedule them
                index[active] = events
        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # my file descriptor event indices
        self._read = collections.defaultdict(list)
        self._write = collections.defaultdict(list)
        self._exception = collections.defaultdict(list)

        # my debug aspect
        import journal
        self._debug = journal.debug('pyre.ipc.selector')

        # all done
        return


    # implementation details
    # private types
    class _event:
        """Encapsulate a channel and the associated call-back"""

        def __init__(self, channel, handler):
            self.channel = channel
            self.handler = handler
            return

        __slots__ = ('channel', 'handler')

    # private data
    _watching = True # controls whether to continue monitoring the event sources


# end of file
