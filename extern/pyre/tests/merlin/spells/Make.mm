# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = merlin-tests

TEST_DIR = /tmp

PROJ_CLEAN += \
    $(TEST_DIR)/merlin.deep \
    $(TEST_DIR)/merlin.one \
    $(TEST_DIR)/merlin.shallow \
    $(TEST_DIR)/merlin.two \

MERLIN = $(EXPORT_BINDIR)/merlin

#--------------------------------------------------------------------------
#

all: test

test: init clean

init:
	$(PYTHON) $(MERLIN) init $(TEST_DIR)/merlin.shallow
	$(PYTHON) $(MERLIN) init $(TEST_DIR)/merlin.one $(TEST_DIR)/merlin.two
	$(PYTHON) $(MERLIN) init --create-prefix $(TEST_DIR)/merlin.deep/ly/burried

# end of file
