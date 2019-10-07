# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre
PROJ_CLEAN += \
    local-make sample.zip

# standard targets
all: test clean

test: sanity virtual local zip explorers clean

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./exceptions.py
	${PYTHON} ./node.py
	${PYTHON} ./folder.py
	${PYTHON} ./filesystem.py
	${PYTHON} ./directory_walker.py
	${PYTHON} ./stat_recognizer.py

virtual:
	${PYTHON} ./virtual.py
	${PYTHON} ./virtual_leaks.py
	${PYTHON} ./virtual_insert.py
	${PYTHON} ./virtual_insert_multiple.py
	${PYTHON} ./virtual_insert_badNode.py
	${PYTHON} ./virtual_find.py
	${PYTHON} ./virtual_subscripts.py
	${PYTHON} ./virtual_access.py
	${PYTHON} ./virtual_info.py

local:
	${PYTHON} ./local.py
	${PYTHON} ./local_leaks.py
	${PYTHON} ./local_find.py
	${PYTHON} ./local_open.py
	${PYTHON} ./local_rootNonexistent.py
	${PYTHON} ./local_rootNotDirectory.py
	${PYTHON} ./local_make.py

zip: scratch
	${PYTHON} ./zip.py
	${PYTHON} ./zip_open.py
	${PYTHON} ./zip_rootNonexistent.py
	${PYTHON} ./zip_rootNotZipfile.py

explorers:
	${PYTHON} ./finder.py
	${PYTHON} ./finder_pattern.py
	${PYTHON} ./simple_explorer.py
	${PYTHON} ./tree_explorer.py

scratch:
	@zip -q sample.zip *

# end of file
