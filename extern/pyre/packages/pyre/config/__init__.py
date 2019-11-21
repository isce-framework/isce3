# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package contains the implementation of the readers and writers of the various
configuration file formats supported by pyre.
"""


# factories
def newCommandLineParser(**kwds):
    """
    Build a new parser of command line arguments
    """
    # access the factory
    from .CommandLineParser import CommandLineParser
    # build one and return it
    return CommandLineParser(**kwds)


def newConfigurator(**kwds):
    """
    Build a new processor of configuration information
    """
    # access the factory
    from .Configurator import Configurator
    # build one and return it
    return Configurator(**kwds)


# end of file
