# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pyre # the framework
import weakref
# my protocol
from .Shell import Shell as shell


# declaration
class Executive(pyre.component, family='pyre.shells.executive', implements=shell):
    """
    The base class for hosting strategies
    """


    # user configurable state
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


    # access to what the framework knows about the runtime environment
    @property
    def host(self):
        """
        Encapsulation of what is known about the runtime environment
        """
        # ask the executive
        return self.pyre_executive.host


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the application behavior
        """
        # {Executive} is abstract
        raise NotImplementedError("class {.__name__!r} must implement 'launch'".format(type(self)))


# end of file
