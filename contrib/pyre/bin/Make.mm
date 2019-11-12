# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# the package name
PACKAGE = bin
# externals
include Python/default.def

# the files
EXPORT_BINS = \
    abi.py \
    blame \
    cache_tag.py \
    class.pyre \
    colors.py \
    dir.py \
    merlin \
    pyre \
    pyre-config \
    python.pyre \
    smith.pyre \
    walk \

# add these to the clean pile
PROJ_TIDY += python.pyre
PROJ_CLEAN = ${addprefix $(EXPORT_BINDIR)/, $(EXPORT_BINS)}

# the standard targets
all: export

export:: $(EXPORT_BINS) export-binaries tidy

live: live-bin

python.pyre: python.cc
	$(CXX) $(CXXFLAGS) $< -o $@ $(LCXXFLAGS) -l$(PYTHON_LIB)

# archiving support
zipit:
	cd $(EXPORT_ROOT); zip -r $(PYRE_ZIP) ${addprefix bin/, $(EXPORT_BINS)}

# end of file
