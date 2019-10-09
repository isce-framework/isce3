# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import functools
# my base class
from .Peer import Peer


# declaration
class Crew(Peer, family="pyre.nexus.peers.crew"):
    """
    The base facilitator of asynchronous task execution in foreign processes, and one of the
    building blocks of process based concurrency in pyre.

    Crew members are typically instantiated by a {recruiter} in matching pairs that are
    connected to each other via bidirectional {channels}. One member of the pair participates
    in the event logic of the host application; this instance is referred to as the team side
    crew member. The other is hosted by a remote pyre {shell} with its own event loop, and is
    referred to as the worker side. Making the worker side functional typically involves
    spinning up a new process, but this considered a {recruiter} implementation detail. The
    pair of crew instances are responsible only for the babysitting of the task execution.

    The team side crew member acts as a proxy for the worker side. The host application
    schedules the execution of a {task} by invoking the team side interface. The crew instance
    serializes the task and sends it off to its remote twin for execution, monitors progress,
    and reports the task result back to the host application.
    """

    # types
    from .exceptions import RecoverableError
    from .CrewStatus import CrewStatus as crewcodes
    from .TaskStatus import TaskStatus as taskcodes


    # interface - team side
    def join(self, team):
        """
        Join a team

        This is invoked by my recruiter on the team side and it is part of team assembly. The
        intent is to make crew members available to the team after they have reported ready to
        receive tasks for execution
        """
        # schedule the handler of the worker side registration
        self.dispatcher.whenReadReady(
            channel = self.channel,
            call = functools.partial(self.activate, team=team))
        # all done
        return self


    def activate(self, channel, team):
        """
        My worker twin is reporting ready to work

        N.B.: this is an event handler; careful with its return value
        """
        # check it's me we are talking about
        assert channel is self.channel
        # get the status of my twin
        status = self.marshaler.recv(channel=channel)
        # and if all is good
        if status is self.crewcodes.healthy:
            # let the team know
            team.activate(crew=self)
            # and add me to the execution schedule
            team.schedule(crew=self)
        # do not reschedule this handler
        return False


    def execute(self, team, task):
        """
        Send my twin the {task} to be executed
        """
        # send the task
        self.marshaler.send(channel=self.channel, item=task)
        # schedule the harvesting of the result
        self.dispatcher.whenReadReady(
            channel = self.channel,
            call = functools.partial(self.assess, team=team, task=task))
        # all done
        return self


    def assess(self, channel, team, task, **kwds):
        """
        Harvest the task completion status
        """
        # grab the report
        memberstatus, taskstatus, result = self.marshaler.recv(channel=channel)
        # show me on the debug channel
        self.debug.log('{me.pid}: {member}, {task}, {result}'.format(
            me=self, member=memberstatus, task=taskstatus, result=result))

        # first, let's figure out what to do with the task; if it failed due to some temporary
        # condition
        if taskstatus is self.taskcodes.failed:
            # tell me
            self.reportRecoverableError(team=team, task=task, error=result)
            # put the task back in the workplan
            team.workplan.add(task)

        # now, let's figure out what to do with me; if i'm healthy
        if memberstatus is self.crewcodes.healthy:
            # put me back in the work queue
            team.schedule(crew=self)
        # otherwise
        else:
            # tell me
            self.reportUnrecoverableError(team=team, task=task, error=result)
            # dismiss me
            team.dismiss(crew=self)

        # all done
        return False


    def dismissed(self):
        """
        My team manager has dismissed me
        """
        # send the end-of-tasks marker
        self.marshaler.send(channel=self.channel, item=None)
        # clean up
        self.resign()
        # leave a note
        self.debug.log('{me.pid}: dismissed at {me.finish:.3f}'.format(me=self))
        # all done
        return self


    def reportRecoverableError(self, team, task, error):
        """
        Report a task failure that can be reasonably expected to be temporary
        """
        # show me
        self.debug.log('{me.pid}: recoverable error: {error}'.format(me=self, error=error))
        # all done
        return


    def reportUnrecoverableError(self, team, task, error):
        """
        Report a permanent task failure
        """
        # show me
        self.debug.log('{me.pid}: unrecoverable error: {error}'.format(me=self, error=error))
        # all done
        return


    # interface - worker side
    def register(self):
        """
        Initialize the worker side
        """
        # send in my registration when the write side of my channel is ready to accept data
        self.dispatcher.whenWriteReady(channel=self.channel, call=self.checkin)
        # and chain up to start processing events
        return self


    def checkin(self, channel):
        """
        Send my team registration now that my communication channel is open
        """
        # check it's me we are talking about
        assert channel is self.channel
        # send in a healthy status code
        self.marshaler.send(channel=channel, item=self.crewcodes.healthy)
        # register the task execution handler
        self.dispatcher.whenReadReady(channel=self.channel, call=self.perform)
        # do not reschedule this handler
        return False


    def perform(self, channel, **kwds):
        """
        A notification has arrived that indicates there is a task waiting to be executed
        """
        # extract the task from the channel
        task = self.marshaler.recv(channel=channel)
        # leave a note
        self.debug.log('{me.pid}: got {task}'.format(me=self, task=task))
        # if it's a quit marker
        if task is None:
            # we are all done
            self.stop()
            # don't reschedule this handler
            return False

        # otherwise, try to
        try:
            # execute the task and collect its result
            result = self.engage(task=task, **kwds)
        # if the task failure is recoverable
        except self.RecoverableError as error:
            # prepare a report with an error code for the task
            taskstatus = self.taskcodes.failed
            # a clean bill of health for me
            crewstatus = self.crewcodes.healthy
            # and attach the error description
            result = error
        # if anything else goes wrong
        except Exception as error:
            # prepare a report with an error code for the task
            taskstatus = self.taskcodes.aborted
            # mark me as damaged
            crewstatus = self.crewcodes.damaged
            # and attach the error description
            result = error
        # if all goes well
        else:
            # indicate task success
            taskstatus = self.taskcodes.completed
            # and a clean bill of health for me
            crewstatus = self.crewcodes.healthy

        # schedule the reporting of the execution of this task
        self.dispatcher.whenWriteReady(
            channel = channel,
            call = functools.partial(self.report,
                                     result = result,
                                     crewstatus = crewstatus,
                                     taskstatus = taskstatus))

        # and go back to waiting for more
        return True


    def engage(self, task, **kwds):
        """
        Carry out the task
        """
        # just do it
        return task(**kwds)


    def report(self, channel, crewstatus, taskstatus, result, **kwds):
        """
        Post the task completion {report}
        """
        # make a report
        report = (crewstatus, taskstatus, result)
        # tell me
        self.debug.log('{me.pid}: sending report {report}'.format(me=self, report=report))
        # serialize and send
        self.marshaler.send(channel=channel, item=report)
        # all done; don't reschedule
        return False


    def resign(self):
        # record my finish time; don't mess with the timer too much as it might not belong to me
        self.finish = self.timer.lap()
        # close my communication channel
        self.channel.close()
        # all done
        return self


    # meta-methods
    def __init__(self, pid, channel, **kwds):
        # chain up
        super().__init__(**kwds)
        # save my crew is; this is an opaque type, assigned to me by my recruiter
        self.pid = pid
        # save the communication channel to my twin
        self.channel = channel
        # all done
        return


# end of file
