# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import enum


# declaration
class CrewStatus(enum.Enum):
    """
    Indicators used by crew members to describe what happened during the execution of the most
    recent task that was assigned to them
    """

    # as far as the crew member can tell, all is good and it is ready to accept work
    healthy = 0
    # the crew member is compromised and can't be relied upon any more; it should be removed
    # from the team permanently
    damaged = 1


# end of file
