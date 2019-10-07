# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis, leif strand
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre
import operator
from time import time as now


# declaration
class Scheduler(pyre.component, family='pyre.ipc.dispatchers.scheduler'):
    """
    Support for invoking event handlers at specified times

    Clients create alarms by invoking {alarm} and supplying an event handler to be invoked and
    specifying the number of seconds before the alarm comes due. The time interval is expected
    to be a dimensional quantity with units of time.

    The current implementation converts the time interval before the alarm comes due into an
    absolute time, and pairs it with the handler into an {_alarm} instance. The {_alarm} is
    then stored into a list of such {_alarms}, and the list is sorted in reverse order,
    i.e. with the alarm that is due next at the end of the list.
    """


    # constants
    from pyre.units.SI import second


    # interface
    @pyre.export
    def alarm(self, interval, call):
        """
        Schedule {call} to be invoked after {interval} elapses.

        parameters:
           {call}: a function that takes the current time and returns a reschedule interval
           {interval}: a dimensional quantity from {pyre.units} with units of time
        """
        # create a new alarm instance
        alarm = self._alarm(time=now()+interval/self.second, handler=call)
        # add it to my list
        self._alarms.append(alarm)
        # sort
        self._alarms.sort(key=operator.attrgetter('time'), reverse=True)
        # and return
        return


    def poll(self):
        """
        Compute the number of seconds until the next alarm comes due.

        If there are no scheduled alarms, {poll} returns {None}; if alarms are overdue, it
        returns 0. This slightly strange logic is designed to satisfy the requirements for
        calling {select}.
        """
        # the necessary information is in the last entry in my {_alarms}, since they are
        # always in descending order
        try:
            # try to grab it
            alarm = self._alarms[-1]
        # if there is nothing there
        except IndexError:
            # we have no scheduled alarms
            return None
        # if it succeeded
        else:
            # get the scheduled time
            due = alarm.time
        # return the number of seconds until it comes due, bound from below
        return max(0, due - now())


    def awaken(self):
        """
        Raise all overdue alarms by calling the registered handlers
        """
        # get my alarms
        alarms = self._alarms
        # initialize the reschedule pile
        reschedule = []
        # get the time
        time = now()

        # iterate through my alarms
        while 1:
            # attempt
            try:
                # to grab one
                alarm = alarms.pop()
            # if none are left
            except IndexError:
                # we are all done raising alarms
                break
            # if this alarm is not due yet
            if time < alarm.time:
                # put it back at the end of the list
                alarms.append(alarm)
                # no need to look any further
                break
            # otherwise, this alarm is overdue; invoke the handler
            delta = alarm.handler(timestamp=time)
            # if the handler indicated that it wants to reschedule this alarm
            if delta:
                # save it
                reschedule.append((delta, alarm.handler))

        # if there is nothing to reschedule
        if not reschedule:
            # all done
            return

        # otherwise, get a fresh timestamp
        time = now()
        # go through the pile
        for interval, call in reschedule:
            # create a new alarm instance
            alarm = self._alarm(time=time+interval/self.second, handler=call)
            # add it to my list
            self._alarms.append(alarm)
        # sort
        self._alarms.sort(key=operator.attrgetter('time'), reverse=True)

        # all done
        return


    # meta methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)
        # the list of alarms; kept sorted in descending order by alarm time, i.e. with the next
        # alarm to go off at the end of the list
        self._alarms = []
        # all done
        return


    # implementation details
    # private types
    class _alarm:
        """Encapsulate the time and event handler of an alarm"""

        def __init__(self, time, handler):
            self.time = time
            self.handler = handler
            return

        def __str__(self): return "alarm: {.time}".format(self)

        __slots__ = ('time', 'handler')


    # private data
    _alarms = None


# end of file
