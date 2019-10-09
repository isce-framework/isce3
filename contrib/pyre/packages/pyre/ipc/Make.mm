# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# access the project defaults
include pyre.def
# the package name
PACKAGE = ipc
# python packages
EXPORT_PYTHON_MODULES = \
    Channel.py \
    Dispatcher.py \
    Marshaler.py \
    Pickler.py \
    Pipe.py \
    Port.py \
    PortTCP.py \
    Scheduler.py \
    Selector.py \
    Socket.py \
    SocketTCP.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
