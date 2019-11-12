# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# support
import pyre


# declaration
class Recruiter(pyre.protocol, family='pyre.nexus.recruiters'):
    """
    The protocol of resource allocation strategies
    """

    # interface
    @pyre.provides
    def recruit(self, team, **kwds):
        """
        Recruit members for the {team}
        """

    @pyre.provides
    def deploy(self, team, **kwds):
        """
        Create a new {team} member
        """

    @pyre.provides
    def dismiss(self, team, member, **kwds):
        """
        The {team} manager has dismissed the given {member}
        """


    # default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default {Recruiter} implementation
        """
        # the default strategy is to create child processes on the local machine
        from .Fork import Fork
        # return the component factory
        return Fork


# end of file
