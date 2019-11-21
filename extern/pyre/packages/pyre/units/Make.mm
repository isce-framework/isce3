# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = units
# the python modules
EXPORT_PYTHON_MODULES = \
    Dimensional.py \
    Parser.py \
    SI.py \
    angle.py \
    area.py \
    density.py \
    energy.py \
    force.py \
    length.py \
    mass.py \
    power.py \
    pressure.py \
    speed.py \
    substance.py \
    temperature.py \
    time.py \
    volume.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
