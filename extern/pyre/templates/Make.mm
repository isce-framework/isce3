# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the templates
TEMPLATES = \
    class.c++ \
    class.python \
    django \
    plexus \
# the destination
TEMPLATE_DIR = $(EXPORT_ROOT)/templates
# standard targets
all: export

export::
	@find . -name \*~ -delete
	@$(RM_RF) $(TEMPLATE_DIR)
	@$(MKDIR) $(MKPARENTS) $(TEMPLATE_DIR)
	@$(CP_R) $(TEMPLATES) $(TEMPLATE_DIR)

LIVE_TEMPLATES = ${addprefix $(PROJ_LIVE_TPLDIR),$(TEMPLATES)}
live:
	$(SSH) $(PROJ_LIVE_USERURL) '$(MKDIRP) $(PROJ_LIVE_TPLDIR)'
	$(SSH) $(PROJ_LIVE_USERURL) '$(RM_RF) $(LIVE_TEMPLATES)'
	$(SCP) -r $(TEMPLATES) $(PROJ_LIVE_USERURL):$(PROJ_LIVE_TPLDIR)

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) templates

# end of file
