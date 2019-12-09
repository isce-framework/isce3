# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# exceptions
from ..framework.exceptions import FrameworkError


# local anchor
class NexusError(FrameworkError):
    """
    Base exceptions for all error conditions detected by nexus components
    """


# a temporary error
class RecoverableError(NexusError):
    """
    A recoverable error has occurred
    """

# connection reset by peer
class ConnectionResetError(NexusError):
    """
    The connection was closed by the peer
    """


# end of file
