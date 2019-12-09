# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import os
import pyre # the framework
from .Executive import Executive # my superclass


# declaration
class Fork(Executive, family='pyre.shells.fork'):
    """
    A shell that invokes the main application behavior in a child process
    """


    # public state
    capture = pyre.properties.bool(default=True)
    capture.doc = "control whether to create communication channels to the daemon process"


    # interface
    @pyre.export
    def launch(self, application, *args, **kwds):
        """
        Invoke the {application} behavior in a subprocess and return a pair of channels
        corresponding to {stdout} and {stderr} of the child, with the write end of the channels
        both connected to the child's {stdin}
        """
        # if we are in debug mode, launch the application
        if application.DEBUG: return application.main(*args, **kwds)

        # build the three pipes
        pipes = self.openCommunicationPipes()
        # and fork
        pid = os.fork()

        # in the parent process
        if pid > 0:
            # build and return the parent side channels
            channels = self.parentChannels(pipes)
            # invoke the parent behavior
            status = application.launched(*args, channels=channels, **kwds)
            # and return its status
            return status

        # in the child process, convert {stdout} and {stderr} into channels
        channels = self.childChannels(pipes)
        # set up a net
        try:
            # launch the application
            status = application.main(*args, channels=channels, **kwds)
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
        # and exit
        raise SystemExit(status)


    # implementation details
    def openCommunicationPipes(self):
        """
        Build three pipes for parent/child communication
        """
        # bail out if we are not supposed to build pipes to the child
        if not self.capture: return (None, None, None)

        # otherwise, build the pipes
        stdin = os.pipe()
        stdout = os.pipe()
        stderr = os.pipe()
        # and return them
        return (stdin, stdout, stderr)


    def parentChannels(self, pipes):
        """
        Build the parent side of the communication channels
        """
        # bail out if we are not supposed to build pipes to the child
        if not self.capture: return (None, None)

        # otherwise, access the pipe factory
        import pyre.ipc
        # unpack
        stdin, stdout, stderr = pipes
        # turn {stdout} and {stderr} into channels
        # careful to identify the read/write ends correctly
        stdout = pyre.ipc.pipe(descriptors=(stdout[0], stdin[1]))
        stderr = pyre.ipc.pipe(descriptors=(stderr[0], stdin[1]))
        # return them
        return stdout, stderr


    def childChannels(self, pipes):
        """
        Build the child side of the communication channels
        """
        # bail out if we are not supposed to build pipes to the child
        if not self.capture: return (None, None)

        # otherwise, access the pipe factory
        import pyre.ipc
        # unpack
        stdin, stdout, stderr = pipes
        # convert {stdout} and {stderr} into channels
        # careful to identify the read/write ends correctly
        stdout = pyre.ipc.pipe(descriptors=(stdin[0], stdout[1]))
        stderr = pyre.ipc.pipe(descriptors=(stdin[0], stderr[1]))
        # and return them
        return stdout, stderr


# end of file
