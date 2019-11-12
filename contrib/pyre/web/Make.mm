# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#

# project defaults
include pyre.def
# the package
PACKAGE = web
# my subfolders with static assets
RECURSE_DIRS = \
    bin \
    graphics \
    styles \

# layout
PROJ_TMPDIR = $(BLD_TMPDIR)/$(PROJECT)/$(PACKAGE)
PROJ_WEBPACK_CONFIG = config
PROJ_WEBPACK_SOURCES = react
PROJ_CLEAN += ${addprefix $(PROJ_TMPDIR)/, build $(PROJ_WEBPACK_SOURCES)}

# the exported items
EXPORT_WEB = \
   $(PROJ_TMPDIR)/build/*

# standard targets
all: webpack.deps webpack.build export

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

export:: export-web
	BLD_ACTION="export" $(MM) recurse

live: live-web
	BLD_ACTION="live" $(MM) recurse

zipit:

# end of file
