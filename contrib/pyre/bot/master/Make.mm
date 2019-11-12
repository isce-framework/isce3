# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre
PACKAGE = bot

BUILDBOT_MASTER=root@pyre.orthologue.com
BUILDBOT_HOME=/var/lib/buildbot/masters/pyre

all: tidy

install:
	scp master.cfg $(BUILDBOT_MASTER):$(BUILDBOT_HOME)

deploy: install
	ssh $(BUILDBOT_MASTER) 'service buildmaster restart'

# end of file
