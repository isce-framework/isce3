#include <complex> // std::complex, std::arg
#include <cstring> // std::memcpy
#include <exception> // std::domain_error

#include "ICU.h" // ICU, isce::io::Raster, size_t, uint8_t

namespace isce { namespace unwrap { namespace icu {

void ICU::unwrap(
    isce::io::Raster & unw,
    isce::io::Raster & ccl,
    isce::io::Raster & intf,
    isce::io::Raster & corr,
    unsigned int seed)
{
    // Raster dims
    const size_t length = intf.length();
    const size_t width = intf.width();
    
    // Buffers for single tile from each input, output Raster
    const size_t bufsize = _NumBufLines * width;
    auto intftile = new std::complex<float>[bufsize];
    auto corrtile = new float[bufsize];
    auto unwtile = new float[bufsize];
    auto ccltile = new uint8_t[bufsize];

    // Wrapped phase
    auto phase = new float[bufsize];

    // Residue charges and neutrons
    auto charge = new signed char[bufsize];
    auto neut = new bool[bufsize];

    // Branch cuts
    auto tree = new bool[bufsize];

    // Current connected component
    auto currcc = new bool[bufsize];

    // Bootstrap lines (unwrapped phase and connected component labels)
    const size_t bssize = _NumBsLines * width;
    auto bsunw = new float[bssize];
    auto bslabels = new uint8_t[bssize];

    // Table of connected component label equivalences
    auto labelmap = LabelMap();

    // Number of lines to next tile
    const size_t step = _NumBufLines - _NumOverlapLines;

    // Number of tiles
    int ntiles = 1;
    if (length > _NumBufLines)
    {
        if (step <= 0)
        {
            throw std::domain_error("number of overlap lines must be less than number of buffer lines");
        }
        ntiles = (length + step-1) / step;
        if (length % step <= _NumOverlapLines) { --ntiles; }
    }

    // Loop over tiles.
    for (int t = 0; t < ntiles; ++t)
    {
        // Read interferogram, correlation lines.
        size_t startline = t * step;
        size_t tilelen = std::min(_NumBufLines, length - startline);
        intf.getBlock(intftile, 0, startline, width, tilelen);
        corr.getBlock(corrtile, 0, startline, width, tilelen);

        // Compute wrapped phase.
        size_t tilesize = tilelen * width;
        for (size_t i = 0; i < tilesize; ++i) { phase[i] = std::arg(intftile[i]); }

        // Get residue charges.
        getResidues(charge, phase, tilelen, width);

        // Generate neutrons to guide the tree-growing process.
        genNeutrons(neut, intftile, corrtile, tilelen, width);

        // Grow trees (make branch cuts).
        growTrees(tree, charge, neut, tilelen, width, seed);

        // Grow grass (find connected components and unwrap phase). If not first 
        // tile, bootstrap phase from previous tile.
        if (t == 0)
        {
            growGrass<false>(
                unwtile, ccltile, currcc, bsunw, bslabels, labelmap, phase, 
                tree, corrtile, _InitCorrThr, tilelen, width);
        }
        else
        {
            growGrass<true>(
                unwtile, ccltile, currcc, bsunw, bslabels, labelmap, phase, 
                tree, corrtile, _InitCorrThr, tilelen, width);
        }

        // If not last tile, get bootstrap data for processing next tile.
        if (t < ntiles-1)
        {
            // Offset to first bootstrap line from start of tile
            size_t bsoff = (_NumBufLines -_NumOverlapLines/2 - _NumBsLines/2) * width;

            // Copy bootstrap lines.
            std::memcpy(bsunw, &unwtile[bsoff], _NumBsLines * width * sizeof(float));
            std::memcpy(bslabels, &ccltile[bsoff], _NumBsLines * width * sizeof(uint8_t));
        }

        // Write out unwrapped phase, connected component labels.
        unw.setBlock(unwtile, 0, startline, width, tilelen);
        ccl.setBlock(ccltile, 0, startline, width, tilelen);
    }

    // If all label mappings are identity, then each connected component is 
    // labelled properly and we are finished. Otherwise, go back and merge 
    // redundant labels.
    bool doUpdateLabels = false;
    for (uint8_t l = 1; l < labelmap.size(); ++l)
    {
        if (labelmap.getlabel(l) != l) 
        { 
            doUpdateLabels = true; 
            break;
        }
    }

    if (doUpdateLabels)
    {
        // Loop over tiles.
        for (int t = 0; t < ntiles; ++t)
        {
            // Read connected component labels.
            size_t startline = t * step;
            size_t tilelen = std::min(_NumBufLines, length - startline);
            ccl.getBlock(ccltile, 0, startline, width, tilelen);

            // Update labels.
            size_t tilesize = tilelen * width;
            for (size_t i = 0; i < tilesize; ++i)
            {
                if (ccltile[i] != 0)
                {
                    ccltile[i] = labelmap.getlabel(ccltile[i]);
                }
            }

            // Write out updated labels.
            ccl.setBlock(ccltile, 0, startline, width, tilelen);
        }
    }

    delete[] intftile;
    delete[] corrtile;
    delete[] unwtile;
    delete[] ccltile;
    delete[] phase;
    delete[] charge;
    delete[] neut;
    delete[] tree;
    delete[] bsunw;
    delete[] bslabels;
}

} } }

