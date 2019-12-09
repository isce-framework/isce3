# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre
# my user configurable state
from .Recruiter import Recruiter


# declaration
class Team(pyre.protocol, family='pyre.nexus.teams'):
    """
    The specification for a process collective that coöperate to carry out a work plan
    """


    # user configurable state
    size = pyre.properties.int()
    size.doc = 'the number of team members to recruit'

    recruiter = Recruiter()
    recruiter.doc = 'the strategy for recruiting team members'


    # interface
    @pyre.provides
    def assemble(self, workplan, **kwds):
        """
        Recruit a team to execute the set of tasks in my {workplan}
        """

    @pyre.provides
    def vacancies(self):
        """
        Compute how may recruits are needed to take the team to full strength
        """


    # my default
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default {Team} implementation
        """
        # use a distributed pool of processes
        from .Pool import Pool
        # so make its component factory available
        return Pool


# end of file
