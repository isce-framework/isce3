# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# access to my base class
import pyre


# declaration
class Shell(pyre.protocol, family="pyre.shells"):
    """
    The protocol implemented by the pyre application hosting strategies
    """


    # public data
    home = pyre.properties.str(default=None)
    home.doc = "the process home directory"

    # machine layout
    hosts = pyre.properties.int(default=1)
    hosts.doc = "the number of hosts in the parallel machine"

    tasks = pyre.properties.int(default=1)
    tasks.doc = "the number of tasks per host"

    gpus = pyre.properties.int(default=0)
    gpus.doc = "the number of GPU coprocessors per task"

    # a marker that enables applications to deduce the type of shell that is hosting them
    model = pyre.properties.str(default='unknown')
    model.doc = "the programming model"


    # my default implementation
    @classmethod
    def pyre_default(cls, **kwds):
        """
        The default shell implementation
        """
        # use {Script}
        from .Script import Script
        return Script


    # interface
    @pyre.provides
    def launch(self, application, *args, **kwds):
        """
        Launch the application
        """


# end of file
