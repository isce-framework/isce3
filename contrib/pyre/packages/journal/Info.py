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
class Info(Diagnostic, Channel):
    """
    This class is the implementation of the info channel
    """

    # public data
    severity = "info"

    # class private data
    _index = collections.defaultdict(Channel.Enabled)


# end of file
