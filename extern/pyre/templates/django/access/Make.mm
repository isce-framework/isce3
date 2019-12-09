# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include {project.name}.def
# the name of this package
PACKAGE = access
# add this to the clean pile
PROJ_CLEAN += authorized_keys
# the list of public keys
PUBLIC_KEYS = $(wildcard *.pub)

# standard targets
all: tidy

# do nothing by default
live:

# make the authorized keys file
authorized_keys: $(PUBLIC_KEYS) grant.py grant.pfg Make.mm
	./grant.py

deploy: authorized_keys
	$(SCP) $< $(PROJ_LIVE_ADMIN)@$(PROJ_LIVE_HOST):$(PROJ_LIVE_HOME)/.ssh

# convenience (and for checking before deploying)
keys: authorized_keys

# end of file
