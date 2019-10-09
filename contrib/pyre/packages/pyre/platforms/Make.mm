# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = platforms
# the python modules
EXPORT_PYTHON_MODULES = \
    Bare.py \
    CPUInfo.py \
    CentOS.py \
    DPkg.py \
    Darwin.py \
    Debian.py \
    Host.py \
    Linux.py \
    MacPorts.py \
    Managed.py \
    Modules.py \
    POSIX.py \
    PackageManager.py \
    Platform.py \
    RedHat.py \
    Ubuntu.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
