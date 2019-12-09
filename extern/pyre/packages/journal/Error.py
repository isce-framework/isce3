# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# packages
import collections


# super-classes
from .Channel import Channel
from .Diagnostic import Diagnostic


# declaration
class Error(Diagnostic, Channel):
    """
    This class is the implementation of the error channel
    """

    # types
    from .exceptions import ApplicationError

    # public data
    severity = "error"

    # class private data
    _index = collections.defaultdict(Channel.Enabled)


    # interface
    def log(self, message=None, stackdepth=0):
        """
        Make a journal entry and build an exception ready to be raised by the caller
        """
        # first, record the entry
        super().log(message, stackdepth)
        # build an instance of the error exception
        error = self.ApplicationError(error=self)
        # don't raise it; let the caller decide what to do with it
        return error


    # class private data
    stackdepth = Diagnostic.stackdepth + 1 # there is an extra stack level for errors...


# end of file
