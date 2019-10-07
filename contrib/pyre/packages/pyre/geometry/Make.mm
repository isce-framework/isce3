# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project defaults
include pyre.def
# package name
PACKAGE = geometry
# the python modules
EXPORT_PYTHON_MODULES = \
    CPGrid.py \
    Field.py \
    Grid.py \
    Mesh.py \
    Octree.py \
    Point.py \
    PointCloud.py \
    Simplex.py \
    SimplicialMesh.py \
    Surface.py \
    Triangulation.py \
    __init__.py

# standard targets
all: export

export:: export-package-python-modules

live: live-package-python-modules

# end of file
