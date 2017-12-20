# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
PROJECT =  gtest
# the package name
PACKAGE = internal/custom

# the top level headers
EXPORT_PKG_HEADERS = \
    gtest-port.h \
    gtest-printers.h \
    gtest.h

all: export

export:: export-package-headers

# end of file
