# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


"""
This package provides support for composing pyre applications

The pyre framework encourages factoring applications into two distinct parts: a subclass of the
component {Application} that defines the runtime behavior of the application, and a hosting
strategy that defines the runtime environment in which the application executes. The hosting
strategy is a subclass of the component {Shell} and is responsible for creating the execution
context for the application.

The package provides a number of ready to use hosting strategies.

{Script} expects an instance of an {Application} subclass, invokes its {main} method, and exits
after handing the value returned to the operating system as the process exit code. It the pyre
equivalent of the familiar launching of executables written in low level languages.

{Daemon} is suitable for applications that run in the background, without access to a
terminal. It performs the steps necessary to detach a process from its parent so that the
parent may exit without causing the child to terminate as well.

{Service} builds on {Daemon} to enable distributed applications by exposing the application
component registry to the network.

Other packages leverage these building blocks to provide support for other hosting
strategies. For a sophisticated example, see the {mpi} package, which provides support for
running concurrent applications using {MPI}.
"""

# the marker of component factories
from .. import foundry

# command support
from .Action import Action as action

# command implementations
@foundry(implements=action)
def command():
    """
    The command base component
    """
    # grab the component class record
    from .Command import Command as command
    # and return it
    return command

@foundry(implements=action)
def panel():
    """
    The command panel base component
    """
    # grab the component class record
    from .Panel import Panel as panel
    # and return it
    return panel


# application hosting support
from .Shell import Shell as shell

# the hosting strategies
@foundry(implements=shell)
def script():
    """
    The basic application shell
    """
    # grab the component class record
    from .Script import Script as script
    # and return it
    return script

@foundry(implements=shell)
def interactive():
    """
    An application shell based on {script} that enters interactive mode after the application
    is finished running
    """
    # grab the component class record
    from .Interactive import Interactive as interactive
    # and return it
    return interactive

@foundry(implements=shell)
def ipython():
    """
    An application shell based on {script} that enters interactive mode after the application
    is finished running
    """
    # grab the component class record
    from .IPython import IPython as ipython
    # and return it
    return ipython

@foundry(implements=shell)
def fork():
    """
    The fork shell: a shell that invokes the application main entry point in a child process
    """
    # grab the component class record
    from .Fork import Fork as fork
    # and return it
    return fork

@foundry(implements=shell)
def daemon():
    """
    The daemon shell: a shell that invokes the application main entry point as long lived
    independent process that has detached completely from its parent
    """
    # grab the component class record
    from .Daemon import Daemon as daemon
    # and return it
    return daemon

@foundry(implements=shell)
def web():
    """
    The web shell: an interactive shell that presents the user with an initial web page
    """
    # grab the component class record
    from .Web import Web as web
    # and return it
    return web


# terminal support
from .Terminal import Terminal as terminal

@foundry(implements=terminal)
def ansi():
    """
    A terminal that supports color control using ANSI escpae sequences
    """
    # grab the component class record
    from .ANSI import ANSI as ansi
    # and return it
    return ansi

@foundry(implements=terminal)
def plain():
    """
    A plain terminal with no special capabilities
    """
    # grab the component class record
    from .Plain import Plain as plain
    # and return it
    return plain


# the base application components
from .Application import Application as application
from .Plexus import Plexus as plexus
# and its support
from .Layout import Layout as layout

# the user component
from .User import User as user


# end of file
