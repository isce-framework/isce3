# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# factories
from .Chain import Chain as chain
from .Command import Command as command
from .File import File as file
from .FileRegion import FileRegion as region
from .NameLookup import NameLookup as lookup
from .Script import Script as script
from .Simple import Simple as simple
from .Tracker import Tracker as tracker


# constants
# the stack depth of the caller relative to {traceback.extract_stack}, taking into account the
# frame added by {pyre} itself
callerStackDepth = 2


# in case we just don't know
def unknown():
    return simple(source='<unknown>')


# dynamic locators
def here(level=0):
    """
    Build a locator that records the caller's location

    The parameter {level} specifies the level above the caller that is to be used as the
    originating location. The default, {level}=0, indicates to use the caller's location;
    setting {level} to 1 will use the caller's caller's location, and so on.
    """
    # externals
    import traceback
    # get a stack trace
    trace = traceback.extract_stack(limit=callerStackDepth+level)
    # grab the information from the current frame
    source, line, function, text = trace[0]
    # hand to the script locator
    return script(source=source, line=line, function=function)


# end of file
