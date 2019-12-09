# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = pyre

MPI_ARGS = --hostfile localhost

all: test

test: sanity groups communications launch

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./extension.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./extension.py
	${PYTHON} ./world.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./world.py

groups:
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./group.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 7 ${PYTHON} ./group_include.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 7 ${PYTHON} ./group_exclude.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 7 ${PYTHON} ./group_setops.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 7 ${PYTHON} ./restrict.py

communications:
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./bcast.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./sum.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./product.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./max.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 8 ${PYTHON} ./min.py
	${MPI_EXECUTIVE} ${MPI_ARGS} -np 7 ${PYTHON} ./port.py

launch: mpirun slurm

mpirun:
	${PYTHON} ./mpirun.py

slurm:
	${PYTHON} ./slurm.py

# end of file
