# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project defaults
include pyre.def


all: test

test: sanity channels scheduler selector clean

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./pickler.py
	${PYTHON} ./pipe.py
	${PYTHON} ./tcp.py

channels:
	${PYTHON} ./pickler_over_pipe.py
	${PYTHON} ./pickler_over_tcp.py

scheduler:
	${PYTHON} ./scheduler.py
	${PYTHON} ./scheduler_instantiation.py
	${PYTHON} ./scheduler_alarms.py

selector:
	${PYTHON} ./selector.py
	${PYTHON} ./selector_instantiation.py
	${PYTHON} ./selector_alarms.py
	${PYTHON} ./selector_pickler_over_pipe.py
	${PYTHON} ./selector_pickler_over_tcp.py


# end of file
