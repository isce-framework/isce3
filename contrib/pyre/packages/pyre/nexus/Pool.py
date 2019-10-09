# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import functools
# support
import pyre
# base class
from .Peer import Peer
# my protocol
from .Team import Team
# my user configurable state
from .Recruiter import Recruiter


# declaration
class Pool(Peer, family='pyre.nexus.teams.pool', implements=Team):
    """
    A process collective that coöperate to carry out a work plan
    """


    # types
    from .Crew import Crew as crew


    # user configurable state
    size = pyre.properties.int(default=1)
    size.doc = 'the number of crew members to recruit'

    recruiter = Recruiter()
    recruiter.doc = 'the strategy for recruiting crew members'


    # interface
    @pyre.export
    def assemble(self, workplan, **kwds):
        """
        Assemble a team to execute the set of tasks in my {workplan}
        """
        # grab a journal channel
        channel = self.debug
        # show me
        channel.line('executing the workplan')
        channel.line('  current outstanding tasks: {}'.format(len(self.workplan)))
        channel.line('  max team size: {}'.format(self.size))
        channel.line('  current vacancies: {}'.format(self.vacancies()))
        channel.line('  registered crew members: {}'.format(len(self.registered)))
        channel.line('  active crew members: {}'.format(len(self.active)))

        # add the new tasks to the workplan
        self.workplan |= workplan
        # tell me
        channel.line('extending the workplan')
        channel.line('  current outstanding tasks: {}'.format(len(self.workplan)))

        # if necessary, recruit some new crew members
        self.recruit()
        # tell me
        channel.line('recruited new crew members')
        channel.line('  registered crew members: {}'.format(len(self.registered)))
        channel.line('  active crew members: {}'.format(len(self.active)))

        # flush
        channel.log()

        # all done
        return self


    @pyre.export
    def vacancies(self):
        """
        Compute how may recruits are needed to take the team to full strength
        """
        # get the current team size
        current = len(self.registered) + len(self.active)
        # get my pool size limit
        pool = self.size
        # figure out how much work is left to do
        tasks = len(self.workplan)

        # compute the number of vacancies
        return min(tasks, pool) - current


    # meta-methods
    def __init__(self, crew=None, **kwds):
        # chain up
        super().__init__(**kwds)

        # if i were given a non-trivial crew factory
        if crew is not None:
            # attach it
            self.crew = crew

        # initialize my crew registries
        self.registered = set()
        self.active = set()
        self.retired = set()

        # my workplan is the set of tasks that are pending
        self.workplan = set()

        # all done
        return


    # implementation details
    def recruit(self, **kwds):
        """
        Assemble the team
        """
        # get my recruiter to recruit some workers
        for crew in self.recruiter.recruit(team=self, **kwds):
            # register the crew member
            self.registered.add(crew)
        # all done
        return self


    def activate(self, crew):
        """
        Add the given {crew} member to the scheduling queue
        """
        # upgrade its status from registered
        self.registered.remove(crew)
        # to active
        self.active.add(crew)
        # all done
        return self


    def schedule(self, crew):
        """
        Add the given {crew} member to the execution schedule
        """
        # start sending tasks when the worker is ready to listen
        self.dispatcher.whenWriteReady(
            channel = crew.channel,
            call = functools.partial(self.submit, crew=crew))
        # all done
        return self


    def submit(self, channel, crew, **kwds):
        """
        A crew member has reported ready to accept tasks
        """
        # N.B.: {channel} is ready to write, because that's how we got here; so write away...

        # tell me
        self.debug.log('sending a task to {.pid}'.format(crew))
        # get my workplan
        workplan = self.workplan
        # and my marshaler
        marshaler = self.marshaler

        # if there is nothing left to do
        if not workplan:
            # notify this worker we are done
            self.dismiss(crew=crew)
            # and don't send it any further work
            return False

        # otherwise, grab a task
        task = workplan.pop()
        # and send it to the worker
        crew.execute(team=self, task=task)

        # don't reschedule me; let the handler that harvests the task status decide the fate of
        # this worker
        return False


    def dismiss(self, crew):
        """
        Dismiss the {crew} member from the team
        """
        # notify this crew member it is dismissed
        crew.dismissed()
        # let the recruiter know
        self.recruiter.dismiss(team=self, crew=crew)
        # remove it from the roster
        self.active.discard(crew)
        # and add it to the pile of retired workers
        self.retired.add(crew)
        # all done
        return self


    # private data
    active = None   # the set of currently deployed crew members
    retired = None  # the set of retired crew members


# end of file
