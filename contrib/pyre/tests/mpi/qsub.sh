#!/bin/bash
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#
#PBS -j oe
#PBS -N ip
#PBS -V
#PBS -l nodes=3:ppn=4
#PBS -l walltime=0:10:00
#PBS -q gpu

# discover the number of processors
nprocs=$(wc -l < ${PBS_NODEFILE})
# go to the working directory
cd ${PBS_O_WORKDIR}
# execute
mpirun python3.3 ip.py
