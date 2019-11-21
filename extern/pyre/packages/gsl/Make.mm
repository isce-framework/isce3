# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include gsl.def
# package name
PACKAGE = gsl
# the python modules
EXPORT_PYTHON_MODULES = \
    Histogram.py \
    Matrix.py \
    MatrixView.py \
    Permutation.py \
    RNG.py \
    Vector.py \
    VectorView.py \
    blas.py \
    linalg.py \
    pdf.py \
    stats.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-python-modules

live: live-python-modules

# end of file
