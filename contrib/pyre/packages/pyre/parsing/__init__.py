# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides support for writing simple parsers
"""


# factories
from .Descriptor import Descriptor as token
from .Scanner import Scanner as scanner
from .SWScanner import SWScanner as sws
from .Parser import Parser as parser


# end of file
