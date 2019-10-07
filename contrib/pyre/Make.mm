# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


# project global settings
include pyre.def
# my subdirectories
RECURSE_DIRS = \
    lib \
    packages \
    extensions \
    defaults \
    bin \
    templates \
    schema \
    etc \
    tests \
    examples \
    bot \
    people

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

# other targets
build: lib packages extensions defaults bin templates web

test: build tests examples

# the pyre install archives
PYTHON_TAG = ${shell $(PYTHON) bin/cache_tag.py}
PYTHON_ABITAG = ${shell $(PYTHON) bin/abi.py}
PYRE_VERSION = $(PROJECT_MAJOR).$(PROJECT_MINOR)
PYRE_BOOTPKGS = pyre journal merlin
PYRE_ZIP = $(EXPORT_ROOT)/pyre-$(PYRE_VERSION).$(PYTHON_ABITAG).zip
PYRE_BOOTZIP = $(EXPORT_ROOT)/pyre-$(PYRE_VERSION)-boot.zip

zip: build cleanit zipit pushit cleanit

cleanit:
	$(RM_F) $(PYRE_ZIP)

zipit:
	for x in bin lib packages defaults templates web; do { \
            (cd $$x; PYRE_ZIP=$(PYRE_ZIP) $(MM) zipit) \
        } done

pushit:
	scp $(PYRE_ZIP) $(PROJ_LIVE_USERURL):$(PROJ_LIVE_DOCROOT)

boot:
	@$(RM_F) $(PYRE_BOOTZIP)
	@(cd $(EXPORT_ROOT)/packages; zip -r ${PYRE_BOOTZIP} $(PYRE_BOOTPKGS) --include \*.py)
	scp $(PYRE_BOOTZIP) $(PROJ_LIVE_USERURL):$(PROJ_LIVE_DOCROOT)
	@$(RM_F) $(PYRE_BOOTZIP)

# shortcuts for building specific subdirectories
.PHONY: $(RECURSE_DIRS) doc

$(RECURSE_DIRS) doc:
	(cd $@; $(MM))


# end of file
