# -*- cmake -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#

# handle the portinfo file
pyre_portinfo()
# build libjournal
pyre_journalLib()
# buld libpyre
pyre_pyreLib()
# and the mpi layers
pyre_mpiLib()

# end of file
