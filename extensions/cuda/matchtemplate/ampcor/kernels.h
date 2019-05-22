// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// michael a.g. aïvázis <michael.aivazis@para-sim.com>
// parasim
// (c) 1998-2019 all rights reserved
//

// code guard
#if !defined(ampcor_libampcor_cuda_correlators_kernels_h)
#define ampcor_libampcor_cuda_correlators_kernels_h

// cuda
#include <cuComplex.h>
// STL
#include <complex>

// forward declarations
namespace ampcor {
    namespace cuda {
        namespace kernels {
            // compute amplitudes of the tile pixels
            void detect(const std::complex<float> * cArena, std::size_t cells, float * rArena);

            // subtract the tile mean from each reference pixel
            void refStats(float * rArena,
                          std::size_t pairs, std::size_t refDim, std::size_t cellsPerTilePair,
                          float * stats);

            // build the sum area tables for the target tiles
            void sat(const float * rArena,
                     std::size_t pairs,
                     std::size_t refCells, std::size_t tgtCells, std::size_t tgtDim,
                     float * sat);

            // compute the average amplitude for all possible placements of a reference shape
            // within the search windows
            void tgtStats(const float * sat,
                          std::size_t pairs,
                          std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
                          float * stats);

            // compute the correlation matrix
            void correlate(const float * rArena, const float * refStats, const float * tgtStats,
                           std::size_t pairs,
                           std::size_t refCells, std::size_t tgtCells, std::size_t corCells,
                           std::size_t refDim, std::size_t tgtDim, std::size_t corDim,
                           float * dCorrelation);

            // compute the locations of the maximum value of the correlation map
            void maxcor(const float * cor,
                        std::size_t pairs, std::size_t corCells, std::size_t corDim,
                        int * loc);

            // nudge the (row, col) pairs so that they describe sub-tiles within a target tile
            void nudge(std::size_t pairs,
                       std::size_t refDim,  std::size_t tgtDim, std::size_t margin,
                       int * locations);

            // migrate the expanded maxcor tiles to the refinement arena
            void migrate(const std::complex<float>  * arena,
                         std::size_t pairs,
                         std::size_t refDim, std::size_t tgtDim, std::size_t expDim,
                         std::size_t refRefinedDim, std::size_t tgtRefinedDim,
                         const int * locations,
                         std::complex<float> * refinedArena);

            // upcast the correlation matrix into complex numbers and embed in the zoomed
            // hyper-matrix
            auto r2c(const float * gamma,
                     std::size_t pairs, std::size_t corDim, std::size_t zmdDim
                     ) -> cuComplex *;

            // convert the zoomed correlation matrix to floats
            auto c2r(const cuComplex * scratch,
                     std::size_t pairs, std::size_t zmdDim
                     ) -> float *;

            // assemble the offset field
            void offsetField(const int * maxcor, const int * zoomed,
                             std::size_t pairs,
                             std::size_t margin, std::size_t refineMargin, std::size_t zoom,
                             float * field);
        }
    }
}

// code guard
#endif

// end of file
