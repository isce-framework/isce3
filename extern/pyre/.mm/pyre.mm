# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# project meta-data
pyre.major := $(repo.major)
pyre.minor := $(repo.minor)
pyre.micro := $(repo.micro)
pyre.revision := $(repo.revision)

# pyre builds a python package
pyre.packages := pyre.pkg journal.pkg mpi.pkg
# libraries
pyre.libraries := journal.lib pyre.lib mpi.lib
# python extensions
pyre.extensions := journal.ext host.ext timers.ext mpi.ext
# and test suites
pyre.tests := mpi.tst.libmpi # pyre.tst.pyre pyre.tst.libpyre

# if we have {cuda}, add it to the pile
${if ${findstring cuda,$(extern.available)},\
    ${eval pyre.packages += cuda.pkg} \
    ${eval pyre.extensions += cuda.ext} \
    ${eval cuda.libraries += cudart} \
,}

# if we have {gsl}, add it to the pile
${if ${findstring gsl,$(extern.available)},\
    ${eval pyre.packages += gsl.pkg} \
    ${eval pyre.extensions += gsl.ext} \
,}

# if we have {libpq}, add it to the pile
${if ${findstring libpq,$(extern.available)}, \
    ${eval pyre.extensions += postgres.ext} \
,}

# the pyre meta-data
pyre.pkg.root := packages/pyre/
pyre.pkg.stem := pyre
pyre.pkg.drivers := pyre pyre-config colors.py dir.py merlin smith.pyre
pyre.pkg.config := pyre merlin
pyre.pkg.ext := extensions/

# the pyre meta-data
journal.pkg.root := packages/journal/
journal.pkg.stem := journal
journal.pkg.ext :=

# the journal library meta-data
journal.lib.root := lib/journal/
journal.lib.stem := journal
journal.lib.incdir := $(builder.dest.inc)pyre/journal/
journal.lib.master := journal.h
journal.lib.extern :=
journal.lib.c++.defines += PYRE_CORE
journal.lib.c++.flags += $($(compiler.c++).std.c++17)

# the extension meta-data
journal.ext.root := extensions/journal/
journal.ext.stem := journal
journal.ext.pkg := journal.pkg
journal.ext.wraps := journal.lib
journal.ext.capsule :=
journal.ext.extern := journal.lib python
journal.ext.lib.c++.defines += PYRE_CORE
journal.ext.lib.c++.flags += $($(compiler.c++).std.c++17)

# the pyre library meta-data
pyre.lib.root := lib/pyre/
pyre.lib.stem := pyre
pyre.lib.extern := journal.lib
pyre.lib.prerequisites += journal.lib
pyre.lib.c++.flags += $($(compiler.c++).std.c++17)

# the pyre extensions
# host info
host.ext.root := extensions/host/
host.ext.stem := host
host.ext.pkg := pyre.pkg
host.ext.wraps := pyre.lib
host.ext.capsule :=
host.ext.extern := pyre.lib journal.lib python
host.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
host.ext.lib.prerequisites += journal.lib # pyre.lib is added automatically
# postgres
postgres.ext.root := extensions/postgres/
postgres.ext.stem := postgres
postgres.ext.pkg := pyre.pkg
postgres.ext.wraps := pyre.lib
postgres.ext.capsule :=
postgres.ext.extern := pyre.lib journal.lib libpq python
postgres.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
postgres.ext.lib.prerequisites += journal.lib # pyre.lib is added automatically
# timers
timers.ext.root := extensions/timers/
timers.ext.stem := timers
timers.ext.pkg := pyre.pkg
timers.ext.wraps := pyre.lib
timers.ext.capsule :=
timers.ext.extern := pyre.lib journal.lib python
timers.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
timers.ext.lib.prerequisites += journal.lib # pyre.lib is added automatically

# cuda
# the package
cuda.pkg.root := packages/cuda/
cuda.pkg.stem := cuda
cuda.pkg.meta :=
cuda.pkg.ext :=
# the extension
cuda.ext.root := extensions/cuda/
cuda.ext.stem := cuda
cuda.ext.pkg := cuda.pkg
cuda.ext.wraps :=
cuda.ext.capsule :=
cuda.ext.extern := pyre.lib journal.lib cuda mpi python
cuda.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
cuda.ext.lib.prerequisites += journal.lib pyre.lib # gsl.lib is added automatically

# gsl
# the package
gsl.pkg.root := packages/gsl/
gsl.pkg.stem := gsl
gsl.pkg.meta :=
gsl.pkg.ext :=
# the extension
gsl.ext.root := extensions/gsl/
gsl.ext.stem := gsl
gsl.ext.pkg := gsl.pkg
gsl.ext.wraps :=
gsl.ext.capsule.destination := pyre/gsl/
gsl.ext.extern := pyre.lib journal.lib mpi.lib gsl mpi python numpy
gsl.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
gsl.ext.lib.prerequisites += journal.lib pyre.lib mpi.ext # gsl.lib is added automatically
# the tests
gsl.tst.pkg.stem := gsl
gsl.tst.pkg.prerequisites := gsl.ext
gsl.tst.pkg.root := tests/gsl/

# mpi
# the package
mpi.pkg.root := packages/mpi/
mpi.pkg.stem := mpi
mpi.pkg.ext :=
# the library
mpi.lib.root := lib/mpi/
mpi.lib.stem := mpi
mpi.lib.extern := journal.lib mpi
mpi.lib.master := mpi.h
mpi.lib.incdir := $(builder.dest.inc)pyre/mpi/
mpi.lib.prerequisites += journal.lib
mpi.lib.c++.flags += $($(compiler.c++).std.c++17)
# the extension
mpi.ext.root := extensions/mpi/
mpi.ext.stem := mpi
mpi.ext.pkg := mpi.pkg
mpi.ext.wraps := mpi.lib
mpi.ext.extern := pyre.lib journal.lib mpi python
mpi.ext.lib.c++.flags += $($(compiler.c++).std.c++17)
mpi.ext.lib.prerequisites += journal.lib pyre.lib # mpi.lib is added automatically
# the tests
mpi.tst.libmpi.stem := mpi
mpi.tst.libmpi.extern := pyre.lib journal.lib mpi
mpi.tst.libmpi.prerequisites := mpi.lib
mpi.tst.libmpi.root := tests/libpyre/mpi/

# the libpyre test suite
pyre.tst.libpyre.stem := libpyre
pyre.tst.libpyre.extern := pyre.lib pyre
pyre.tst.libpyre.prerequisites += pyre.lib

# the pyre package test suite
pyre.tst.pyre.stem := pyre
pyre.tst.pyre.prerequisites += pyre.pkg pyre.ext

# end of file
