# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# bootstrapping
# the current default builds of openmpi on linux do not link their pluggins against libmpi
# so they all report unresolved symbols; the temporary fix is to change the way python
# dlopens our extension module so the mpi symbols go to the global namespace, where the
# pluggins will be able to find them. hopefully, the openmpi people will fix this soon
# LAST CHECKED: 20120423, revision 1402, openmpi 1.4.3
import sys
if sys.platform.startswith('linux'):
    # that's where the stupid flag value lives...
    import ctypes
    # adjust the {dlopen} flags
    sys.setdlopenflags(sys.getdlopenflags() | ctypes.RTLD_GLOBAL)

# we start out with {world} being a trivial commuincator
from .TrivialCommunicator import TrivialCommunicator as world

# attempt to load the mpi extension
try:
    # try to load the extension
    from . import mpi
# if it fails for any reason
except Exception as error:
    # indicate that there is no runtime support
    mpi = None
# otherwise, we have bindings and hence MPI support
else:
    # grab the shell protocol form pyre
    from pyre import foundry, shells

    # the foundries for the shells in this package
    # the default shell is raw {mpirun}
    @foundry(implements=shells.shell)
    def mpirun():
        # get the class
        from .Launcher import Launcher
        # and return it
        return Launcher

    # support for SLURM
    @foundry(implements=shells.shell)
    def slurm():
        # get the class
        from .Slurm import Slurm
        # and return it
        return Slurm

    # the factories
    communicator = None
    group = None
    port = None


# initialization
def init():
    """
    Initialize the runtime

    We used to do this automatically, but that's not possible any more. The reason is that
    processes cannot fork/exec {mpirun} after they have called {MPI_Init}. Apparently, this has
    been discouraged always by {openmpi}, and it is explicitly prohibited with {openmpi-3.0}. So
    here we are...
    """
    # if we don't have runtime support
    if not mpi:
        # bail
        return None

    # register the finalization routine to happen when the interpreter exits
    import atexit
    atexit.register(finalize)

    # initialize mpi
    mpi.init()
    # provide access to the extension through the base mpi object
    from .Object import Object
    Object.mpi = mpi

    # the factory placeholders
    global communicator
    global group
    global port
    # the factories
    from .Communicator import Communicator as communicator
    from .Group import Group as group
    from .Port import Port as port

    # fix the world communicator
    global world
    # by building a real one
    world = communicator(capsule=mpi.world)

    # all done
    return mpi

def finalize():
    """
    Shutdown mpi
    """
    # if we don't have runtime support
    if not mpi:
        # bail
        return

    # get the groups
    from .Group import Group
    # ask each instance
    for group in Group._pyre_extent:
        # to forget its capsule
        group.capsule = None

    # get the communicators
    from .Communicator import Communicator
    # ask each instance
    for communicator in Communicator._pyre_extent:
        # to forget its capsule
        communicator.capsule = None

    # get {Object}
    from .Object import Object
    # and destroy the extension handle it holds
    Object.mpi = None

    # finalize
    mpi.finalize()

    # all done
    return


# in any case, grab the framework
import pyre
# register the package
package = pyre.executive.registerPackage(name='mpi', file=__file__)
# record the layout
home, prefix, defaults = package.layout()

# the package meta-data
from . import meta

# administrative
def copyright():
    """
    Return the pyre copyright note
    """
    return print(meta.header)

def license():
    """
    Print the pyre license
    """
    # print it
    return print(meta.license)

def version():
    """
    Return the pyre version
    """
    return meta.version

def credits():
    """
    Print the acknowledgments
    """
    # print it
    return print(meta.acknowledgments)


# end of file
