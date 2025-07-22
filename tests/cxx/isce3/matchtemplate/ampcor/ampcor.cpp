
#include <complex>
#include <gtest/gtest.h>

// support
#include <numeric>
#include <random>
#include <pyre/grid.h>
#include <pyre/journal.h>
// ampcor
#include "isce3/matchtemplate/ampcor/correlators/correlators.h"


// type aliases
// my value type
using value_t = float;
// the pixel type
using pixel_t = std::complex<value_t>;
// my raster type
using slc_t = pyre::grid::simple_t<2, pixel_t>;
// the correlator
using correlator_t = ampcor::correlators::sequential_t<slc_t>;
// adapt a chunk of memory into a tile
using tile_t = pyre::grid::grid_t<slc_t::cell_type,
                                  slc_t::layout_type,
                                  pyre::memory::constview_t<value_t>>;
// adapt a chunk of memory into a tile
using tilec_t = pyre::grid::grid_t<slc_t::cell_type,
                                  slc_t::layout_type,
                                  pyre::memory::constview_t<slc_t::cell_type>>;


// Sanity check. 
// Verify that correlator data space is properly allocated and 
// accessible
TEST(Ampcor, SanityCheck)
{

    // the number of pairs
    correlator_t::size_type pairs = 1;
    // the reference shape
    correlator_t::shape_type refShape = { 64, 64};
    // the search window shape
    correlator_t::shape_type tgtShape = { 80, 80 };

    // make a correlator
    correlator_t c(pairs, refShape, tgtShape);
    // verify that its scratch space is allocated and accessible
    auto arena = c.arena();

    ASSERT_NE(arena, nullptr);
}





// Testing that reference and target tiles get properly loaded to 
// correlator
TEST(Ampcor, AddTiles)
{
    // the reference tile extent
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // therefore, the target tile extent
    int tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    int placements = 2*margin + 1;
    // the number of pairs
    slc_t::size_type pairs = placements*placements;

    // the number of cells in a reference tile
    slc_t::size_type refCells = refDim * refDim;
    // the number of cells in a target tile
    slc_t::size_type tgtCells = tgtDim * tgtDim;
    // the number of cells per pair
    slc_t::size_type cellsPerPair = refCells + tgtCells;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);


    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // compute the pair id
            int pid = i*placements + j;

            // make a reference raster
            slc_t ref(refLayout);
            // fill it with ones
            std::fill(ref.view().begin(), ref.view().end(), pid);

            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with ones
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }



    // verify the reference and target tiles
    // get the arena
    auto arena = c.arena();
    // go through all pairs
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; i<placements; ++i) {
            // compute the pair id
            auto pid = i*placements + j;
            // get the reference raster
            auto ref = arena + pid*cellsPerPair;
            // verify its contents
            for (auto idx=0; idx<refDim; ++idx) {
                for (auto jdx=0; jdx<refDim; ++jdx) {
                    // the expected value
                    pixel_t expected = pid;
                    // the actual value
                    pixel_t actual = ref[idx*refDim + jdx];
                    // Check match
                    ASSERT_EQ(expected, actual);
                }
            }

            // get the target raster
            auto tgt = arena + pid*cellsPerPair + refCells;
            // verify its contents
            for (auto idx=0; idx<refDim; ++idx) {
                for (auto jdx=0; jdx<refDim; ++jdx) {
                    // the bounds of the copy of the ref tile in the tgt tile
                    auto within = (idx >= i && idx < i+refDim && jdx >= j && idx < j+refDim);
                    // the expected value depends on whether we are within the magic subtile
                    pixel_t expected = within ? ref[idx*refDim + jdx] : 0;
                    // the actual value
                    pixel_t actual = tgt[idx*tgtDim + jdx];
                    // Check match
                    ASSERT_EQ(expected, actual);
                }
            }
        }
    }
}




// Testing that conversion of complex to amplitude of correlator dataset is ok
TEST(Ampcor, ComputeAmplitude)
{
    // the reference tile extent
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;

    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells per pair
    auto cellsPerPair = refCells + tgtCells;
    // the total number of cells
    auto cells = pairs * cellsPerPair;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // compute the pair id
            int pid = i*placements + j;

            // make a reference raster
            slc_t ref(refLayout);
            // fill it with the pair id
            std::fill(ref.view().begin(), ref.view().end(), pid);

            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }

    // get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // verify
    for (auto cell=0; cell < cells; ++cell) 
        ASSERT_EQ(std::abs(cArena[cell]), rArena[cell]);

    // clean up
    delete [] rArena;

}



// Testing nudge, i.e., shifting the target tile such that it doesn't extend
// beyond the main footprint
TEST(Ampcor, Nudge)
{
    // the reference tile extent
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // the refinement factor
    int refineFactor = 2;
    // the margin around the refined target tile
    int refineMargin = 4;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    // the number of pairs
    auto pairs = placements*placements;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};
    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout, refineFactor, refineMargin);

    // make an array of locations to simulate the result of {_maxcor}
    int * loc = new int[2*pairs];
    // fill with standard strategy: the (r,c) target tile have a maximum at (r,c)
    for (auto pid = 0; pid < pairs; ++pid) {
        // decode the row and column
        int row = pid / placements;
        int col = pid % placements;
        // record
        loc[2*pid] = row;
        loc[2*pid + 1] = col;
    }

    //main function to test
    c._nudge(loc, refDim, tgtDim);

    // the lowest possible index
    int low = 0;
    // and the highest possible index
    int high = tgtDim - (refDim + 2*refineMargin);
    // go through the locations and check
    for (auto pid = 0; pid < pairs; ++pid) {
        // get the row and column
        auto row = loc[2*pid];
        auto col = loc[2*pid + 1];
        // verify
        ASSERT_LE(low, row);
        ASSERT_LE(low, col);
        ASSERT_GE(high, row);
        ASSERT_GE(high, col);
    }

    // clean up
    delete [] loc;
}





// Testing Sum Area Table
TEST(Ampcor, SAT)
{
    // the reference tile extent
    int refDim = 2;
    // the margin around the reference tile
    int margin = 1;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;

    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells per pair
    auto cellsPerPair = refCells + tgtCells;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<value_t> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random number pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), view.begin());
            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, rview);
            c.addTargetTile(pid, tgt.constview());
        }
    }


    // get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);


    // verify: go through all the tables and check that the lower right hand corner contains
    // the sum of all the tile elements
    for (auto pid = 0; pid < pairs; ++pid) {
        // compute the start of this tile
        auto begin = rArena + pid*cellsPerPair + refDim*refDim;
        // and one past the end of this tile
        auto end = begin + tgtDim*tgtDim;
        // compute the sum
        auto expected = std::accumulate(begin, end, 0.0);
        // get the value form the LRC of the corresponding SAT
        auto computed = sat[(pid+1)*tgtDim*tgtDim - 1];
        // check the difference
        ASSERT_FLOAT_EQ(expected, computed);
    }

    // clean up
    delete [] sat;
    delete [] rArena;

}





 
// Testing reference image statistics computation (for correlation)
TEST(Ampcor, ReferenceStats)
{
    // the reference tile extent
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;

    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in each pair
    auto cellsPerPair = refDim*refDim + tgtDim*tgtDim;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<value_t> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random number pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            slc_t::slice_type slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            slc_t::view_type view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(ref.view().begin(), ref.view().end(), view.begin());

            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }


    // Get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // subtract the tile mean from each pixel in the reference tile and compute the variance
    auto refStats = c._refStats(rArena, refDim, tgtDim);



    // make a lambda that accumulates the square of a value
    auto square = [] (value_t partial, value_t pxl) -> value_t { return partial + pxl*pxl; };
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    // verify
    for (auto pid = 0; pid < pairs; ++pid) {
        // compute the starting address of this tile
        auto tile = rArena + pid * cellsPerPair;
        // compute the mean of the tile
        auto mean = std::accumulate(tile, tile+refCells, 0.0) / refCells;
        // verify it's near zero
        //ASSERT_FLOAT_EQ(std::abs(mean), zeroMean);
        ASSERT_NEAR(std::abs(mean), 0.0, tolerance);

        // compute the variance of the tile
        auto expectedVar = std::sqrt(std::accumulate(tile, tile+refCells, 0.0, square));
        // the computed value
        auto computedVar = refStats[pid];
        // check the mismatch
        ASSERT_FLOAT_EQ(expectedVar, computedVar);
    }

    // clean up
    delete [] refStats;
    delete [] rArena;
}





// Testing target image statistics computation (for correlation)
TEST(Ampcor, TargetStats)
{
    // the reference tile extent
    int refDim = 16;
    // the margin around the reference tile
    int margin = 4;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    //  the dimension of the correlation matrix
    auto corDim = placements;

    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells in the table of mean values
    auto corCells = corDim * corDim;
    // the number of cells in each pair
    auto cellsPerPair = refCells + tgtCells;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<float> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random number pulled from the normal distribution
        ref[idx] = normal(rng);
    }

    // make a view over the reference tile
    auto rview = ref.constview();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto view = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), view.begin());

            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, rview);
            c.addTargetTile(pid, tgt.constview());
        }
    }

    // get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);

    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto tgtStats = c._tgtStats(sat, refDim, tgtDim, corDim);

    // compare expected with computed
    // go through all the pairs
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    for (auto pid = 0; pid < pairs; ++pid) {
        // find the beginning of the target tile
        value_t * tgtStart = rArena + pid*cellsPerPair + refCells;
        // make a tile
        tile_t tgt { tgtLayout, tgtStart };
        // locate the table of mean values for this pair
        value_t * stats = tgtStats + pid*corCells;
        // go through all the placements
        for (auto i=0; i<corDim; ++i) {
            for (auto j=0; j<corDim; ++j) {
                // the offset to the stats for this tile for this placement
                auto offset = i*corDim + j;
                // slice the target tile
                auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
                // make a view
                auto view = tgt.constview(slice);
                // use it to compute the average value in the slice
                auto expectedMean = std::accumulate(view.begin(), view.end(), 0.0) / refCells;
                // read the computed value
                auto computedMean = stats[offset];
                // check mismatch
                ASSERT_NEAR(expectedMean, computedMean, tolerance);
            }
        }
    }

    delete [] tgtStats;
    delete [] sat;
    delete [] rArena;
}



// Testing migration of data from coarse correlation space to refine correlation space
TEST(Ampcor, Migrate)
{
    // the reference tile extent
    int refDim = 32;
    // the margin around the reference tile
    int margin = 8;
    // the refinement factor
    int refineFactor = 2;
    // the margin around the refined target tile
    int refineMargin = 4;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    // the number of pairs
    auto pairs = placements*placements;

    // the number of cells in a reference tile
    auto refCells = refDim * refDim;
    // the number of cells in a target tile
    auto tgtCells = tgtDim * tgtDim;
    // the number of cells per pair
    auto cellsPerPair = refCells + tgtCells;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};
    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // the shape of the refined reference tiles
    auto refRefinedShape = refineFactor * refShape;
    // the shape of the refined target tiles
    auto tgtRefinedShape = refineFactor * (refShape + slc_t::index_type::fill(2*refineMargin));
    // the layout of the refined reference tiles
    slc_t::layout_type refRefinedLayout { refRefinedShape };
    // the layout of the refined target tiles
    slc_t::layout_type tgtRefinedLayout { tgtRefinedShape };
    // the number of cells in a refined reference tile
    auto refRefinedCells = refRefinedLayout.size();
    // the number of cells in a refined target tile
    auto tgtRefinedCells = tgtRefinedLayout.size();
    //  the number of cells per refined pair
    auto cellsPerRefinedPair = refRefinedCells + tgtRefinedCells;

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout, refineFactor, refineMargin);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<float> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random numbers pulled from the normal distribution
        ref[idx] = 1; // normal(rng) + 1if*normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build reference tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto view = tgt.view(slice);
            // place a copy of the reference tile
            std::copy(rview.begin(), rview.end(), view.begin());
            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, ref.constview());
            c.addTargetTile(pid, tgt.constview());
        }
    }

    // push the data to the device
    auto coarseArena = c.arena();


    // allocate a new arena
    auto refinedArena = c._refinedArena();

    // synthesize the locations of the maxima
    // we build a (row, col) index for each pair
    auto locCells = 2 * pairs;
    // allocate
    int * loc = new int[locCells];
    // initialize it
    for (auto pid=0; pid < pairs; ++pid) {
        // decode the row and column and place the values
        loc[2*pid] = pid / placements;
        loc[2*pid + 1] = pid % placements;
    }

    // nudge them
    c._nudge(loc, refDim, tgtDim);

    // migrate
    c._tgtMigrate(coarseArena, loc, refinedArena);


    // Verify
    
    // the dimension of the expanded maxcor slice
    auto expDim = refDim + 2*refineMargin;
    // go through all the pairs
    for (auto pid = 0; pid < pairs; ++pid) {
        // decode the pair id into an index
        // compute the beginning of the target tile in the refined arena
        //auto trmem = refined + pid*cellsPerRefinedPair + refRefinedCells;
        auto trmem = refinedArena + pid*cellsPerRefinedPair + refRefinedCells;
        // build a grid over it
        tilec_t tgtRefined { tgtRefinedLayout, trmem };
        // compute the beginning of the correct target tile
        auto tmem = c.arena() + pid*cellsPerPair + refCells;
        // build a grid over it
        tilec_t tgt { tgtLayout, tmem };
        // find the ULHC of the tile expanded maxcor tile in the target tile
        auto base = tilec_t::index_type {loc[2*pid], loc[2*pid+1]};

        // go through it
        for (auto idx : tgtRefined.layout()) {
            // in the expanded region
            if (idx[0] >= expDim || idx[1] >= expDim) {
                // make sure we have a zero
                ASSERT_FLOAT_EQ(std::abs(tgtRefined[idx]), 0.0);
            } else {
                // what i got
                auto actual = tgtRefined[idx];
                // what i expect
                auto expected = tgt[base+idx];
                // if there is a mismatch
                ASSERT_FLOAT_EQ(actual.real(), expected.real());
                ASSERT_FLOAT_EQ(actual.imag(), expected.imag());
            }
        }
    }

    // clean up
    delete [] loc;
    delete [] refinedArena;
}





// Testing the correlation function
TEST(Ampcor, Correlate)
{
    // the tile type
    typedef pyre::grid::grid_t<value_t, 
                               pyre::grid::layout_t<
                                   pyre::grid::index_t<std::array<int, 4>>>, 
                               pyre::memory::constview_t<value_t>> ctile_t;

    // the reference tile extent
    int refDim = 16;
    // the margin around the reference tile
    int margin = 4;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    //  the dimension of the correlation matrix
    auto corDim = placements;

    // the number of pairs
    auto pairs = placements*placements;

    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<float> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random numbers pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build the target tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto tgtView = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), tgtView.begin());

            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, rview);
            c.addTargetTile(pid, tgt.constview());
        }
    }


    // get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // compute reference tile statistics
    auto refStats = c._refStats(rArena, refDim, tgtDim);

    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);

    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto tgtStats = c._tgtStats(sat, refDim, tgtDim, corDim);

    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto gamma = c._correlate(rArena, refStats, tgtStats, refDim, tgtDim, corDim);

    // the shape of the correlation matrix: the first two indices identify the pair, the last
    // two indicate the placement of the reference tile within the target search window
    ctile_t::shape_type corShape = {corDim, corDim, corDim, corDim};
    // the layout of the correlation matrix
    ctile_t::layout_type corLayout = { corShape };
    // adapt the correlation matrix into a grid
    ctile_t cgrid { corLayout, gamma };

    // verify by checking that the correlation is unity for the correct placement of the
    // reference tile within the target window
    auto tolerance = 10 * std::numeric_limits<value_t>::epsilon();
    for (auto idx = 0; idx < corDim; ++idx) {
        for (auto jdx = 0; jdx < corDim; ++jdx) {
            // we expect
            auto expectedCor = 1.0f;
            // the magic placement should also have unit correlation
            auto computedCor = cgrid[{idx, jdx, idx, jdx}];
            // check if mismatch
            ASSERT_NEAR(expectedCor, computedCor, tolerance);
        }
    }

    // clean up
    delete [] gamma;
    delete [] tgtStats;
    delete [] sat;
    delete [] rArena;
}




// Testing identification of the correlation peak
TEST(Ampcor, MaxCor)
{
    // the tile type
    typedef pyre::grid::grid_t<int, 
                               pyre::grid::layout_t<
                                   pyre::grid::index_t<std::array<int,3>>>, 
                               pyre::memory::constview_t<int>> ctile_t;

    // the reference tile extent
    int refDim = 16;
    // the margin around the reference tile
    int margin = 4;
    // therefore, the target tile extent
    auto tgtDim = refDim + 2*margin;
    // the number of possible placements of the reference tile within the target tile
    auto placements = 2*margin + 1;
    //  the dimension of the correlation matrix
    auto corDim = placements;

    // the number of pairs
    auto pairs = placements*placements;


    // the reference shape
    slc_t::shape_type refShape = {refDim, refDim};
    // the search window shape
    slc_t::shape_type tgtShape = {tgtDim, tgtDim};

    // the reference layout with the given shape and default packing
    slc_t::layout_type refLayout = { refShape };
    // the search window layout with the given shape and default packing
    slc_t::layout_type tgtLayout = { tgtShape };

    // make a correlator
    correlator_t c(pairs, refLayout, tgtLayout);

    // we fill the reference tiles with random numbers
    // make a device
    std::random_device dev {};
    // a random number generator
    std::mt19937 rng { dev() };
    // use them to build a normal distribution
    std::normal_distribution<float> normal {};

    // make a reference raster
    slc_t ref(refLayout);
    // and fill it
    for (auto idx : ref.layout()) {
        // with random numbers pulled from the normal distribution
        ref[idx] = normal(rng);
    }
    // make a view over the reference tile
    auto rview = ref.constview();
    // build the target tiles
    for (auto i=0; i<placements; ++i) {
        for (auto j=0; j<placements; ++j) {
            // make a target tile
            slc_t tgt(tgtLayout);
            // fill it with zeroes
            std::fill(tgt.view().begin(), tgt.view().end(), 0);
            // make a slice
            auto slice = tgt.layout().slice({i,j}, {i+refDim, j+refDim});
            // make a view of the tgt tile over this slice
            auto tgtView = tgt.view(slice);
            // fill it with the contents of the reference tile for this pair
            std::copy(rview.begin(), rview.end(), tgtView.begin());

            // compute the pair id
            int pid = i*placements + j;
            // add this pair to the correlator
            c.addReferenceTile(pid, rview);
            c.addTargetTile(pid, tgt.constview());
        }
    }

    // get a handle on the data
    auto cArena = c.arena();

    // compute the amplitude of every pixel
    auto rArena = c._detect(cArena, refDim, tgtDim);

    // compute reference tile statistics
    auto refStats = c._refStats(rArena, refDim, tgtDim);

    // compute the sum area tables
    auto sat = c._sat(rArena, refDim, tgtDim);

    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto tgtStats = c._tgtStats(sat, refDim, tgtDim, corDim);

    // compute the average amplitude of all possible ref shaped sub-tiles in the target tile
    auto gamma = c._correlate(rArena, refStats, tgtStats, refDim, tgtDim, corDim);

    // compute the locations of the correlation maxima
    auto loc = c._maxcor(gamma, corDim);

    // the shape of the correlation matrix: the first two indices identify the pair, the last
    // two indicate the placement of the reference tile within the target search window
    ctile_t::shape_type shape = { corDim, corDim, 2 };
    // the layout of the correlation matrix
    ctile_t::layout_type layout = { shape };
    // adapt the correlation matrix into a grid
    ctile_t cgrid { layout, loc };
    //ctile_t cgrid { layout, results };

    // verify by checking that the correlation max is at the correct placement of the
    // reference tile within the target window
    for (auto idx = 0; idx < corDim; ++idx) {
        for (auto jdx = 0; jdx < corDim; ++jdx) {
            // the magic placement should also have unit correlation
            auto computedRow = cgrid[{idx, jdx, 0}];
            auto computedCol = cgrid[{idx, jdx, 1}];
            // Check significant mismatch
            ASSERT_EQ(computedRow, idx);
            ASSERT_EQ(computedCol, jdx);
        }
    }

    // clean up
    delete [] loc;
    delete [] gamma;
    delete [] tgtStats;
    delete [] sat;
    delete [] rArena;
}




int main(int argc, char * argv[]) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

