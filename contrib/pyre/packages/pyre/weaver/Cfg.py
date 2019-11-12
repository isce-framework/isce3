# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to the pyre package
import pyre
# my ancestor
from .LineMill import LineMill


# my declaration
class Cfg(LineMill):
    """
    Support for pyre configuration files
    """


    # interface
    def section(self, name):
        """
        Render a new section
        """
        # easy enough
        return "[ {} ]".format(name)


    def trait(self, name, value):
        """
        Render a trait configuration
        """
        # easy enough
        return "{} = {}".format(name, value)


    # private data
    comment = ';'


# end of file
