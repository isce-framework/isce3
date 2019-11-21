# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# the framework
import pyre
# externals
import sys
import subprocess
# my superclass
from pyre.shells.Script import Script


# declaration
class Launcher(Script, family="mpi.shells.mpirun"):
    """
    Encapsulation of launching an MPI job using {mpirun}
    """


    # user configurable state
    hosts = pyre.properties.int(default=1)
    hosts.doc = "the number of hosts in the parallel machine"

    tasks = pyre.properties.int()
    tasks.default = pyre.executive.host.cpus.cores
    tasks.doc = "the number of mpi tasks per host; defaults to the number of cores"

    hostfile = pyre.properties.path()
    hostfile.doc = "the name of the file that describes the machine"

    auto = pyre.properties.bool(default=True)
    auto.doc = "set to {True} to re-launch this script under {mpirun}"

    mpi = pyre.externals.mpi()
    mpi.doc = "the mpi runtime of choice"

    extra = pyre.properties.str()
    extra.doc = "extra arguments to pass to {mpirun}"

    # a marker that enables applications to deduce the type of shell that is hosting them
    model = pyre.properties.str(default="mpi")
    model.doc = "the programming model"


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Launch {application} as a collection of mpi tasks
        """
        # if we are auto-spawning
        if self.auto:
            # first, attempt to
            try:
                # give the application an opportunity to inspect the machine layout
                application.pyre_mpi(*args, **kwds)
            # if it doesn't know how or doesn't care
            except AttributeError:
                # no worries
                pass
            # invoke the launcher
            return self.spawn(application=application, *args, **kwds)

        # otherwise, just launch the application
        return self.parallel(application=application, *args, **kwds)


    # launching hooks; subclasses may override this to get finer control over the two launching
    # branches
    def parallel(self, *args, **kwds):
        """
        Called after the parallel machine has been built and it is time to invoke the user's
        code in every node
        """
        # pull the runtime support
        import mpi
        # and try to initialize it
        if mpi.init():
            # if all goes well, grant access to the global communicator
            self.world = mpi.world
            # launch the application and return its exit code
            return super().launch(*args, **kwds)

        # if something went wrong, get the journal
        import journal
        # make a channel
        channel = journal.error("mpi.init")
        # complain
        channel.log("failed to initialize the mpi runtime support")
        # and bail with an error code
        return 1


    def spawn(self, application, *args, **kwds):
        """
        Invoke {mpirun} with the correct arguments to create the  parallel machine
        """
        # get the command line
        argv = self.buildCommandLine()
        # prep the subprocess options
        options = {
            "args": argv,
            "universal_newlines": True,
            "shell": False
        }

        # if launching fails, indicate an error
        status = 42
        # launch
        with subprocess.Popen(**options) as child:
            # wait for it to finish
            status = child.wait()

        # all done
        return status


    def buildCommandLine(self):
        """
        Construct the mpirun command line
        """
        # the mpi launcher
        launcher = self.mpi.launcher
        # the python interpreter
        interpreter = sys.executable
        # the number of hosts
        hosts = self.hosts
        # the number of tasks per host
        tasks = self.tasks
        # the host file
        hostfile = self.hostfile
        # and the extra command line arguments to {mpirun}
        extra = self.extra

        # start building the command line for the subprocess
        argv = [ launcher ]
        # build the corresponding arguments
        argv += [ "-n", str(hosts * tasks) ]
        # if the user supplied a host file
        if hostfile:
            # add it to the pile
            argv += [ "--hostfile", str(hostfile) ]
        # if the user has anything else to say to mpi
        if extra:
            # add them as well
            argv += extra.split()
        # add the executable
        argv += [ interpreter ]
        # and the entire original command line
        argv += sys.argv
        # be explicit about the machine layout, in case the application made changes
        argv += [ f"--shell.hosts={hosts}", f"--shell.tasks={tasks}" ]
        # prevent auto-spawning next time around
        argv += [ f"--shell.auto=no" ]

        # all done
        return argv


    # easy access to the global communicator; not available until after the launcher has
    # spawned the parallel program
    world = None


# end of file
