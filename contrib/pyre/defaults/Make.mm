# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package
PACKAGE = defaults

# my subfolders
RECURSE_DIRS = \
    merlin \

# the files
EXPORT_ETC = \
    merlin.pfg \
    pyre.pfg

# add these to the clean pile
PROJ_CLEAN += ${addprefix $(EXPORT_ETCDIR)/, $(EXPORT_ETC)}

# the standard targets
all: export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-etc
	BLD_ACTION="export" $(MM) recurse

live: live-etc
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) ${addprefix defaults/, $(EXPORT_ETC) $(RECURSE_DIRS)}

# end of file
