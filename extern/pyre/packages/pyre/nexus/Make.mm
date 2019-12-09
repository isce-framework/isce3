# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# access the project defaults
include pyre.def
# the package name
PACKAGE = nexus
# python packages
EXPORT_PYTHON_MODULES = \
    Asynchronous.py \
    Crew.py \
    CrewStatus.py \
    Fork.py \
    Nexus.py \
    Node.py \
    Peer.py \
    Pool.py \
    Recruiter.py \
    Server.py \
    Service.py \
    Task.py \
    TaskStatus.py \
    Team.py \
    exceptions.py \
    __init__.py

# the standard build targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
