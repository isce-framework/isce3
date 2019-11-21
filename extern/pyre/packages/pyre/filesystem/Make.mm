# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = filesystem
# the python modules
EXPORT_PYTHON_MODULES = \
    BlockDevice.py \
    CharacterDevice.py \
    Directory.py \
    Explorer.py \
    File.py \
    Filesystem.py \
    Finder.py \
    Folder.py \
    HDF5.py \
    Info.py \
    InfoFile.py \
    InfoFolder.py \
    InfoStat.py \
    InfoZip.py \
    InfoZipFolder.py \
    InfoZipFile.py \
    Link.py \
    Local.py \
    Naked.py \
    NamedPipe.py \
    Node.py \
    Recognizer.py \
    SimpleExplorer.py \
    Socket.py \
    Stat.py \
    TreeExplorer.py \
    Walker.py \
    Zip.py \
    exceptions.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
