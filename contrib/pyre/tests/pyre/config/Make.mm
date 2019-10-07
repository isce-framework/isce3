# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#


PROJECT = pyre

#--------------------------------------------------------------------------
#

work:
	${PYTHON} ./configurator_load_pfg.py

all: test

test: sanity events configurator commandline

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py

events:
	${PYTHON} ./events.py
	${PYTHON} ./events_assignments.py

configurator:
	${PYTHON} ./configurator.py
	${PYTHON} ./configurator_assignments.py
	${PYTHON} ./configurator_load_pml.py
	${PYTHON} ./configurator_load_cfg.py
	${PYTHON} ./configurator_load_pfg.py

commandline:
	${PYTHON} ./command.py
	${PYTHON} ./command_argv.py
	${PYTHON} ./command_config.py

# end of file
