# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project defaults
include pyre.def
# package name
PACKAGE = schema
# the files
EXPORT_ETC = \
    config.html \
    config.xsd \

# standard targets
all: tidy

export:: export-etc

live: live-etc

# end of file
