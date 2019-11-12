# -*- Makefile -*-
#
# michael a.g. aïvázis
# orthologue
# (c) 1998-2019 all rights reserved
#


PROJECT = gsl

#--------------------------------------------------------------------------
#

all: test

test: sanity rng pdf vectors matrices blas linalg stats mpi

sanity:
	${PYTHON} ./sanity.py
	${PYTHON} ./rng.py
	${PYTHON} ./pdf.py
	${PYTHON} ./permutation.py
	${PYTHON} ./vector.py
	${PYTHON} ./matrix.py
	${PYTHON} ./blas.py
	${PYTHON} ./linalg.py

rng:
	${PYTHON} ./rng_available.py
	${PYTHON} ./rng_allocate.py
	${PYTHON} ./rng_range.py
	${PYTHON} ./rng_int.py
	${PYTHON} ./rng_float.py

pdf:
	${PYTHON} ./pdf_uniform.py
	${PYTHON} ./pdf_uniform_pos.py
	${PYTHON} ./pdf_gaussian.py
	${PYTHON} ./pdf_dirichlet.py

permutations:
	${PYTHON} ./permutation_allocate.py
	${PYTHON} ./permutation_copy.py
	${PYTHON} ./permutation_get.py

vectors:
	${PYTHON} ./vector_allocate.py
	${PYTHON} ./vector_zero.py
	${PYTHON} ./vector_fill.py
	${PYTHON} ./vector_random.py
	${PYTHON} ./vector_clone.py
	${PYTHON} ./vector_set.py
	${PYTHON} ./vector_slices.py
	${PYTHON} ./vector_contains.py
	${PYTHON} ./vector_tuple.py
	${PYTHON} ./vector_add.py
	${PYTHON} ./vector_sub.py
	${PYTHON} ./vector_mul.py
	${PYTHON} ./vector_div.py
	${PYTHON} ./vector_shift.py
	${PYTHON} ./vector_scale.py
	${PYTHON} ./vector_max.py
	${PYTHON} ./vector_min.py
	${PYTHON} ./vector_minmax.py
	${PYTHON} ./vector_view.py

matrices:
	${PYTHON} ./matrix_allocate.py
	${PYTHON} ./matrix_zero.py
	${PYTHON} ./matrix_fill.py
	${PYTHON} ./matrix_random.py
	${PYTHON} ./matrix_clone.py
	${PYTHON} ./matrix_set.py
	${PYTHON} ./matrix_slices.py
	${PYTHON} ./matrix_contains.py
	${PYTHON} ./matrix_tuple.py
	${PYTHON} ./matrix_add.py
	${PYTHON} ./matrix_sub.py
	${PYTHON} ./matrix_mul.py
	${PYTHON} ./matrix_div.py
	${PYTHON} ./matrix_shift.py
	${PYTHON} ./matrix_scale.py
	${PYTHON} ./matrix_max.py
	${PYTHON} ./matrix_min.py
	${PYTHON} ./matrix_minmax.py
	${PYTHON} ./matrix_view.py

blas:
	${PYTHON} ./blas_ddot.py
	${PYTHON} ./blas_dnrm2.py
	${PYTHON} ./blas_dasum.py
	${PYTHON} ./blas_daxpy.py
	${PYTHON} ./blas_dgemv.py
	${PYTHON} ./blas_dtrmv.py
	${PYTHON} ./blas_dtrsv.py
	${PYTHON} ./blas_dsymv.py
	${PYTHON} ./blas_dgemm.py
	${PYTHON} ./blas_dsymm.py
	${PYTHON} ./blas_dtrmm.py

linalg:
	${PYTHON} ./linalg_LU.py

stats:
	${PYTHON} ./stats_correlation.py
	${PYTHON} ./stats_covariance.py

# do I have mpi?
MPI_DIR=
mpi:
ifneq ($(strip $(MPI_DIR)), )
	${PYTHON} ./matrix_bcast.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./matrix_bcast.py
	${PYTHON} ./matrix_collect.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./matrix_collect.py
	${PYTHON} ./matrix_partition.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./matrix_partition.py
	${PYTHON} ./vector_bcast.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./vector_bcast.py
	${PYTHON} ./vector_collect.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./vector_collect.py
	${PYTHON} ./vector_partition.py
	${MPI_EXECUTIVE} -np 8 ${PYTHON} ./vector_partition.py
endif

# end of file
