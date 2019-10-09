# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# externals
import pickle

# superclasses
from .Object import Object


# declaration
class Port(Object):
    """
    A simple point-to-point communication conduit for a pair of processes
    """


    # per-instance public data
    peer = None # my peer process
    tag = 0 # the default message tag
    communicator = Object.mpi.world # the communicator my peer and I belong to


    # class interface
    # sending and receiving python objects
    def recv(self):
        """
        Receive a python object from my peer
        """
        # get the data
        data = self.mpi.recvBytes(self.communicator.capsule, self.peer, self.tag)
        # extract the object and return it
        return pickle.loads(data)


    def send(self, item):
        """
        Pack and send {item} to my peer
        """
        # pickle the {item}
        data = pickle.dumps(item)
        # and ship it
        return self.mpi.sendBytes(self.communicator.capsule, self.peer, self.tag, data)


    # sending and receiving strings
    def recvString(self):
        """
        Receive a string from my peer
        """
        # pass the buck to the extension module
        return self.mpi.recvString(self.communicator.capsule, self.peer, self.tag)


    def sendString(self, string):
        """
        Send a string to my peer
        """
        # pass the buck to the extension module
        return self.mpi.sendString(self.communicator.capsule, self.peer, self.tag, string)


    # meta methods
    def __init__(self, peer, tag=tag, communicator=communicator, **kwds):
        # chain to my ancestors
        super().__init__(**kwds)

        self.peer = peer
        self.tag = tag
        self.communicator = communicator

        # all done
        return


# end of file
