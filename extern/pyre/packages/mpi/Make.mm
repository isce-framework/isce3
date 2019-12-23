# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include mpi.def
# package name
PACKAGE = mpi
# the python modules
EXPORT_PYTHON_MODULES = \
    Cartesian.py \
    Communicator.py \
    Group.py \
    Launcher.py \
    Object.py \
    Port.py \
    Slurm.py \
    TrivialCommunicator.py \
    meta.py \
    __init__.py

# get today's date
TODAY = ${strip ${shell date -u}}
# grab the revision number
REVISION = ${strip ${shell git log --format=format:"%h" -n 1}}
# if not there
ifeq ($(REVISION),)
REVISION = 0
endif

# standard targets
all: export

export:: meta.py export-python-modules
	@$(RM) meta.py

live: live-python-modules

revision: meta.py export-python-modules
	@$(RM) meta.py

# construct my {meta.py}
meta.py: meta.py.in Make.mm
	@sed \
          -e "s:@MAJOR@:$(PROJECT_MAJOR):g" \
          -e "s:@MINOR@:$(PROJECT_MINOR):g" \
          -e "s:@MICRO@:$(PROJECT_MICRO):g" \
          -e "s:@REVISION@:$(REVISION):g" \
          -e "s|@TODAY@|$(TODAY)|g" \
          meta.py.in > meta.py

# end of file
