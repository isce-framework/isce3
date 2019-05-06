# -*- Makefile -*-
#
# eric m. gurrola
# Jet Propulsion Lab/Caltech
# (c) 2017 all rights reserved
#

# project global settings
include isce.def

# my subdirectories
RECURSE_DIRS = \
    $(PACKAGES)

# the ones that are always available
PACKAGES = \
    attitude \
    datetime \
    doppler \
    ellipsoid \
    interpolator \
    lut \
    matrix \
    cube \
    orbit \
    poly \
    projections \
    serialization \

# the standard targets
all:
	BLD_ACTION="all" $(MM) recurse

tidy::
	BLD_ACTION="tidy" $(MM) recurse

clean::
	BLD_ACTION="clean" $(MM) recurse

distclean::
	BLD_ACTION="distclean" $(MM) recurse

live:
	BLD_ACTION="live" $(MM) recurse

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) ${addprefix packages/, $(PACKAGES) --include \*.py}

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS)

$(RECURSE_DIRS):
	(cd $@; $(MM))


# end of file
