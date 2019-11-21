# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
Definitions for all the exceptions raised by this package
"""


# superclass
from ..framework.exceptions import FrameworkError


# the local base
class RecordError(FrameworkError):
    """
    The base class of all exceptions raised by this package
    """


# something's wrong with an input source
class SourceSpecificationError(RecordError):
    """
    A method that reads records from external input sources was given an invalid input
    specification
    """

    # public data
    description = "invalid input source specification"


# end of file
