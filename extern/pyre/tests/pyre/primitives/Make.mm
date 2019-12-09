# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

PROJ_CLEAN += scratch

# standard targets
all: test clean

test: sanity paths

sanity:
	${PYTHON} ./sanity.py

paths: prep
	${PYTHON} ./path.py
	${PYTHON} ./path_arithmetic.py
	${PYTHON} ./path_parts.py
	${PYTHON} ./path_resolution.py
	${PYTHON} ./path_tuple.py

prep:
	@$(RM_RF) scratch
	@$(MKDIR) scratch
	@$(CD) scratch; \
           $(LN_S) . here; \
           $(LN_S) .. up; \
           $(LN_S) cycle cycle; \
           $(LN_S) $$(pwd)/loop loop; \
           $(LN_S) $$(pwd)/cycle ramp; \
           $(LN_S) tic toc; \
           $(LN_S) toc tic; \


# end of file
