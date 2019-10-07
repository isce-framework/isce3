# -*- Makefile -*-
#
# {project.authors}
# {project.affiliations}
# (c) {project.span} all rights reserved
#

# project defaults
include {project.name}.def
# the package name
PACKAGE = etc/apache

# the standard build targets
all: tidy

live: live-apache-conf live-apache-restart

# end of file
