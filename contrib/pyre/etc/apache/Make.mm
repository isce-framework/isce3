# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project globals
include pyre.def
# the package
PACKAGE = web/apache

# standard targets
all: tidy

live: $(PROJECT).conf live-apache-conf live-apache-restart

# there is another target that might be useful:
#
#    live-apache-conf: make a link to the configuration file in the apache {sites-available}
#                      directory, followed by enabling the site

# assemble the apache configuration file
$(PROJECT).conf: $(PROJECT)_conf Make.mm
	@sed \
          -e "s:PYRE_MAJOR:$(PROJECT_MAJOR):g" \
          -e "s:PYRE_MINOR:$(PROJECT_MINOR):g" \
          -e "s:PROJ_LIVE_HOST:$(PROJ_LIVE_HOST):g" \
          -e "s:PROJ_LIVE_HOME:$(PROJ_LIVE_HOME):g" \
          -e "s:PROJ_LIVE_WEBDIR:$(PROJ_LIVE_WEBDIR):g" \
          -e "s:PROJ_LIVE_DOCROOT:$(PROJ_LIVE_DOCROOT):g" \
          $(PROJECT)_conf > $(PROJECT).conf

# end of file
