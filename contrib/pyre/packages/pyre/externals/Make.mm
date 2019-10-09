# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = externals
# the python modules
EXPORT_PYTHON_MODULES = \
    BLAS.py \
    Cython.py \
    GCC.py \
    GSL.py \
    HDF5.py \
    Installation.py \
    Library.py \
    LibraryInstallation.py \
    Metis.py \
    MPI.py \
    Package.py \
    ParMetis.py \
    PETSc.py \
    Postgres.py \
    Python.py \
    Tool.py \
    ToolInstallation.py \
    VTK.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
