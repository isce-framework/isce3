# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# support
import pyre

# the protocols
from .Nexus import Nexus as nexus
from .Service import Service as service

# the implementations
from .Node import Node as node
from .Server import Server as server

# task support
from .Task import Task as task
from .TaskStatus import TaskStatus as taskcodes
from .CrewStatus import CrewStatus as crewcodes
# task distribution protocols
from .Team import Team as team
from .Recruiter import Recruiter as recruiter
from .Asynchronous import Asynchronous as asynchronous


# task distribution implementations
@pyre.foundry(implements=recruiter, tip="use fork to recruit team members")
def fork():
    """
    Use fork as the team member recruitment strategy
    """
    # get the implementation
    from .Fork import Fork as fork
    # and return it
    return fork


@pyre.foundry(implements=asynchronous, tip="a component that endows a process with an event loop")
def peer():
    """
    The base class for components that provide an application with its event loop
    """
    # get the implementation
    from .Peer import Peer as peer
    # and return it
    return peer


@pyre.foundry(implements=team, tip="a team manager")
def pool():
    """
    The manager of a team that execute a workplan concurrently
    """
    # get the implementation
    from .Pool import Pool as pool
    # and return it
    return pool


# end of file
