# -*- coding: utf-8 -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# attempt to
try:
    # load the extension module
    from . import cuda
# if this fails
except ImportError:
    # not much to do...
    msg = "could not find 'cuda' support"
    # complain
    import journal
    journal.error('cuda').log(msg)
    # re-raise the exception so clients can cope
    raise


# otherwise, all is well;
# pull in the administrivia
version = cuda.version
copyright = cuda.copyright
def license() : print(cuda.license())


# get the exceptions
from . import exceptions
# register the exceptions with the extension module
cuda.registerExceptions(exceptions)


# build the device manager
from .DeviceManager import DeviceManager
manager = DeviceManager()


# end of file
