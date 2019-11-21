# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# meta-class
from pyre.patterns.Singleton import Singleton
# the extension with CUDA support
from . import cuda as libcuda


# declaration
class DeviceManager(metaclass=Singleton):
    """
    The singleton that provides access to what is known about CUDA capable hardware
    """


    # public data
    count = 0
    devices = []


    # interface
    def device(self, did=0):
        """
        Set {did} as the current device
        """
        # delegate to the extension module
        return libcuda.setDevice(did)


    def reset(self):
        """
        Reset the current device
        """
        # easy enough
        return libcuda.resetDevice()


    # meta-methods
    def __init__(self, **kwds):
        # chain up
        super().__init__(**kwds)

        # grab the device class
        from .Device import Device
        # build the device list and attach it
        self.devices = libcuda.discover(Device)
        # set the count
        self.count = len(self.devices)

        # all done
        return


# end of file
