# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2017 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = people
# add this to the clean pile
PROJ_CLEAN += authorized_keys
# the list of public keys
PUBLIC_KEYS = $(wildcard *.pub)

# standard targets
all: tidy
# make the autorized keys file
authorized_keys: $(PUBLIC_KEYS) grant.py grant.pfg Make.mm
	./grant.py

keys: authorized_keys

live: keys
	$(SCP) authorized_keys $(PROJ_LIVE_ADMIN)@$(PROJ_LIVE_HOST):$(PROJ_LIVE_HOME)/.ssh

# end of file
