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
class Firewall(Diagnostic, Channel):
    """
    This class is the implementation of the firewall channel
    """


    # types
    from .exceptions import FirewallError


    # public data
    fatal = False
    severity = "firewall"


    # interface
    def log(self, message=None, stackdepth=0):
        """
        Record my message to my device
        """
        # first, record the entry
        super().log(message, stackdepth)
        # build an instance of the firewall exception
        error = self.FirewallError(firewall=self)
        # if firewalls are not fatal, return the exception instance
        if not self.fatal: return error
        # otherwise, raise it
        raise error


    # class private data
    _index = collections.defaultdict(Channel.Enabled)
    stackdepth = Diagnostic.stackdepth + 1 # there is an extra stack level for firewalls...


# end of file
