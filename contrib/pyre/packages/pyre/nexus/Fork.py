# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
# support
import pyre
# my protocol
from .Recruiter import Recruiter


# declaration
class Fork(pyre.component, family='pyre.nexus.recruiters.fork', implements=Recruiter):
    """
    Create worker processes by cloning the current one
    """


    # protocol obligations
    @pyre.provides
    def recruit(self, team, **kwds):
        """
        Recruit members for the {team}
        """
        # compute the number of vacancies in the team
        vacancies = team.vacancies()
        # recruit the right number of team members
        for _ in range(vacancies):
            # deploy them and add them to the team
            yield self.deploy(team=team, **kwds)
        # all done
        return


    @pyre.provides
    def deploy(self, team, **kwds):
        """
        Create a new {team} member using the {fork} system call
        """
        # team members communicate with the manager using pipes
        parent, child = pyre.ipc.pipe()
        # clone the current process
        pid = os.fork()

        # N.B.: it is important that the worker side of a new team member gets a fresh event
        # loop manager, while the team side is tied to the shared one.

        # in the worker process
        if pid == 0:
            # make a team member
            crew = team.crew(pid=os.getpid(), channel=parent, **kwds)
            # ask it to register with the team
            crew.register()
            # spin up and carry out tasks until there is nothing more to do
            status = crew.run()
            # at which point, this process must terminate
            raise SystemExit(status)

        # make a member proxy for the team manager and return it
        crew = team.crew(pid=pid, channel=child, timer=team.timer)
        # adjust its support for asynchrony
        crew.dispatcher = team.dispatcher
        # and its message serializer
        crew.marshaler = team.marshaler
        # spin it up and return it
        return crew.join(team=team)


    @pyre.provides
    def dismiss(self, team, crew, **kwds):
        """
        The {team} manager has dismissed the given {member}
        """
        # harvest the status
        status = os.waitpid(crew.pid, 0)
        # all done
        return


# end of file
