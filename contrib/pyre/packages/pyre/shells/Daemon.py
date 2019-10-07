# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os # access to os services
import sys
import pyre # access the framework
from .Fork import Fork # my superclass


# declaration
class Daemon(Fork, family="pyre.shells.daemon"):
    """
    A shell that turns a process into a daemon, i.e. a process that is detached from its
    parent and has no access to a terminal
    """


    # user configurable state
    capture = pyre.properties.bool(default=False) # to override the default from {Fork}
    capture.doc = "control whether to create communication channels to the daemon process"
    # a marker that enables applications to deduce the type of shell that is hosting them
    model = pyre.properties.str(default='daemon')
    model.doc = "the programming model"

    daemon = pyre.properties.bool(default=False)
    daemon.doc = "internal marker to indicate that the spawning is complete"


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the application behavior

        For daemons, this is somewhat involved: the process forks twice in order to detach
        itself completely from its parent.
        """
        # if i was told not to spawn, just invoke the behavior
        if application.DEBUG: return application.main(*args, **kwds)

        # if spawning is done
        if self.daemon:
            # figure out where to park the process
            home = self.home or '/'
            # go there
            os.chdir(home)
            # set up a net
            try:
                # launch the application
                status = application.main(*args, **kwds)
            # if anything goes wrong
            except self.PyreError as error:
                # log it
                application.error.log(str(error))
                # set the status
                status = 1
            # if all went well
            else:
                # shutdown
                application.pyre_shutdown(status=status)
            # return the exit code
            raise SystemExit(status)

        # otherwise, build the communication channels
        pipes = self.openCommunicationPipes()
        # fork
        pid = os.fork()

        # in the parent process, build and return the parent side channels
        if pid > 0:
            # create the channels to the child
            channels = self.parentChannels(pipes)
            # invoke the parent behavior
            status = application.launched(*args, channels=channels, **kwds)
            # and return its status
            return status

        # in the intermediate child, decouple from the parent environment; don't change the
        # {cwd} just yet so {exec} below can find the script that launched the application
        os.setsid()
        os.umask(0)

        # respawn
        pid = os.fork()

        # in the intermediate process
        if pid > 0:
            # exit immediately and unconditionally
            return os._exit(0)

        # in the final child process, convert {stdout} and {stderr} into channels
        stdout, stderr = self.childChannels(pipes)

        # daemons use {stdin}, {stdout} and {stderr} to communicate with the launching process
        os.dup2(stdout.inbound, 0)
        os.dup2(stdout.outbound, 1)
        os.dup2(stderr.outbound, 2)

        # on python 3.4 and later
        try:
            # we have to explicitly ask for the file descriptors to become available across
            # calls to {exec}; grab the function that enables descriptor inheritance
            mark = os.set_inheritable
        # older versions don't have this function
        except AttributeError:
            # but also pass file descriptors on to child processes, so no worries
            pass
        # if all is well
        else:
            # mark the descriptors
            mark(0, True)
            mark(1, True)
            mark(2, True)

        # build the command line
        argv = [sys.executable] + sys.argv + ["--{.pyre_name}.daemon=True".format(self)]

        # and try
        try:
            # to exec so that signals get delivered correctly
            os.execv(sys.executable, argv)
        # if anything goes wrong
        except Exception as error:
            # show me
            print(error, file=sys.stderr)

        # all done
        return


# end of file
