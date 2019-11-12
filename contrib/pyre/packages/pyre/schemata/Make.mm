# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = schemata
# the python modules
EXPORT_PYTHON_MODULES = \
    Array.py \
    Boolean.py \
    Catalog.py \
    Complex.py \
    Component.py \
    Container.py \
    Date.py \
    Decimal.py \
    Dimensional.py \
    EnvPath.py \
    EnvVar.py \
    Float.py \
    Fraction.py \
    INet.py \
    InputStream.py \
    Integer.py \
    List.py \
    Mapping.py \
    Numeric.py \
    OutputStream.py \
    Path.py \
    Schema.py \
    Sequence.py \
    Set.py \
    String.py \
    Time.py \
    Timestamp.py \
    Tuple.py \
    Typed.py \
    URI.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
