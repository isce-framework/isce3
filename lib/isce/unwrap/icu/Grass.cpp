#include <algorithm> // std::min
#include <cmath> // round
#include <cstdint> // uint8_t, UINT8_MAX
#include <exception> // std::out_of_range, std::runtime_error

#include "ICU.h" // ICU, LabelMap, idx2_t, offset2_t

namespace isce::unwrap::icu
{

void growConnComp(
    float * unw, 
    bool * currcc, 
    size_t * ccsize, 
    const float * phase, 
    const float * corr, 
    const bool * tree, 
    const idx2_t & seed, 
    const float corrthr, 
    const size_t length, 
    const size_t width)
{
    constexpr float twopi = 2.f * M_PI;

    // Single pixel offsets in each cardinal direction
    constexpr const offset2_t searchpts[4] = {{ 1,  0}, 
                                              { 0, -1},
                                              {-1,  0},
                                              { 0,  1}};

    // Init mask of pixels in the current connected component.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i) { currcc[i] = false; }

    // Maintain list of pixels on perimeter of connected component.
    auto newlist = new idx2_t[tilesize];
    auto oldlist = new idx2_t[tilesize];
    size_t nnew, nold;

    // Start connected component with seed.
    size_t iseed = seed[1] * width + seed[0];
    unw[iseed] = phase[iseed];
    currcc[iseed] = true;
    oldlist[0] = seed;
    *ccsize = nold = 1;

    // Iteratively grow connected component outward from seed, unwrapping phase 
    // and marking pixels that are part of the current component.
    while (nold > 0)
    {
        // Init length of new perimeter list.
        nnew = 0;

        // Loop over existing list of perimeter pixels.
        for (size_t p = 0; p < nold; ++p)
        {
            idx2_t pix = oldlist[p];
            size_t ipix = pix[1] * width + pix[0];
            float phi0 = unw[ipix];

            // Loop over cardinal directions.
            for (int s = 0; s < 4; ++s)
            {
                offset2_t off = searchpts[s];

                // Check for out-of-bounds (avoiding underflow for 
                // unsigned + signed).
                if ((off[0] < 0 && pix[0] < -off[0]) ||
                    (off[0] > 0 && pix[0] + off[0] >= width) ||
                    (off[1] < 0 && pix[1] < -off[1]) ||
                    (off[1] > 0 && pix[1] + off[1] >= length))
                { 
                    continue; 
                }

                // Check if already unwrapped or low-correlation.
                idx2_t newpix = {pix[0] + off[0], pix[1] + off[1]};
                size_t inewpix = newpix[1] * width + newpix[0];
                if (currcc[inewpix] || corr[inewpix] < corrthr) { continue; }

                // Get unwrapped phase and mark as part of current component.
                float phi = phase[inewpix];
                float dphi = twopi * round((phi0 - phi) / twopi);
                unw[inewpix] = phi + dphi;
                currcc[inewpix] = true;

                // If the new perimeter pixel is not on a branch cut, append it 
                // to new list.
                if (!tree[inewpix])
                {
                    newlist[nnew] = newpix;
                    ++nnew;
                }
            }
        }

        // Keep track of connected component size.
        *ccsize += nnew;

        // New list becomes old list (reuse old old list buffer for new new list).
        std::swap(oldlist, newlist);
        nold = nnew;
    }

    delete[] newlist;
    delete[] oldlist;
}

enum BootstrapStatus_t
{
    // Successfully obtained bootstrap phase estimate. Apply phase 
    // bootstrapping.
    BootstrapSuccess = 0,
    // Insufficient overlap in bootstrap region. Don't do phase bootstrapping.
    NoBootstrap,
    // Bootstrap phase variance too high (presumably due to unwrapping errors). 
    // Retry unwrapping with increased correlation threshold.
    BootstrapFailure
};

BootstrapStatus_t estimBootstrapPhase(
    float * bsphase, 
    const float * unw,
    const bool * currcc,
    const float * bsunw,
    const uint8_t * bsccl,
    const size_t width,
    const size_t numBsLines,
    const size_t minBsPts,
    const float bsPhaseVarThr)
{
    // Integrate phase differences (squared) in bootstrap overlap region.
    float sum = 0.f;
    float sumSq = 0.f;
    size_t n = 0;
    for (size_t i = 0; i < numBsLines * width; ++i)
    {
        if (currcc[i] && bsccl[i] != 0)
        {
            float phi = unw[i] - bsunw[i];
            sum += phi;
            sumSq += phi*phi;
            ++n;
        }
    }

    BootstrapStatus_t status;
    if (n < minBsPts)
    {
        status = NoBootstrap;
    }
    else
    {
        // Get mean and variance of phase difference.
        float mu = sum / float(n);
        float Sigma = sumSq / float(n) - (mu * mu);

        if (Sigma < bsPhaseVarThr)
        {
            // Get bootstrap phase (round mean phase difference to nearest 
            // two pi).
            constexpr float twopi = 2.f * M_PI;
            *bsphase = twopi * round(mu / twopi);
            status = BootstrapSuccess;
        }
        else
        {
            status = BootstrapFailure;
        }
    }
    return status;
}

uint8_t bootstrapLabel(
    LabelMap & labelmap,
    const bool * currcc,
    const uint8_t * bsccl, 
    const size_t width, 
    const size_t numBsLines)
{
    // Get the min label among connected components in the bootstrap overlap 
    // region.
    uint8_t minlabel = UINT8_MAX;
    const size_t bssize = numBsLines * width;
    for (size_t i = 0; i < bssize; ++i)
    {
        if (currcc[i] && bsccl[i] != 0)
        {
            minlabel = std::min(minlabel, labelmap.getlabel(bsccl[i]));
        }
    }

    // If there are any connected components in the bootstrap overlap region 
    // with different labels, update their label mapping.
    for (size_t i = 0; i < bssize; ++i)
    {
        if (currcc[i] && bsccl[i] != 0)
        {
            uint8_t oldlabel = labelmap.getlabel(bsccl[i]);
            if (oldlabel != minlabel)
            {
                labelmap.setlabel(oldlabel, minlabel);
            }
        }
    }

    return minlabel;
}

template<bool DO_BOOTSTRAP>
void ICU::growGrass(
    float * unw,
    uint8_t * ccl,
    bool * currcc,
    float * bsunw,
    uint8_t * bsccl, 
    LabelMap & labelmap,
    const float * phase, 
    const bool * tree, 
    const float * corr, 
    float corrthr,
    const size_t length,
    const size_t width)
{
    // Make sure bootstrap lines are not out-of-range of tile.
    if (DO_BOOTSTRAP && length < _NumOverlapLines/2 + _NumBsLines/2)
    {
        throw std::out_of_range("bootstrap lines out-of-range");
    }

    // Offset to first bootstrap line from start of tile
    const size_t bsoff = (_NumOverlapLines/2 - _NumBsLines/2) * width;

    // Init unwrapped phase and connected component labels.
    const size_t tilesize = length * width;
    for (size_t i = 0; i < tilesize; ++i)
    {
        unw[i] = 0.f;
        ccl[i] = 0;
    }

    // Loop over 2D grid of seeds (make sure at least one row & col of seeds 
    // is placed).
    const size_t seedColSpcng = std::min(_MinBsPts, width);
    const size_t seedRowSpcng = std::min(_NumBsLines, length);
    for (size_t sj = seedRowSpcng/2; sj < length; sj += seedRowSpcng)
    {
        for (size_t si = seedColSpcng/2; si < width; si += seedColSpcng)
        {
            // Skip seed if on branch cut, low-correlation, or already 
            // unwrapped.
            idx2_t seed = {si, sj};
            size_t iseed = sj * width + si;
            if (tree[iseed] || corr[iseed] < corrthr || ccl[iseed] != 0) { continue; }

            // Grow connected component from seed.
            size_t ccsize;
            growConnComp(
                unw, currcc, &ccsize, phase, corr, tree, seed, corrthr, length, 
                width);

            // Check if connected component is large enough.
            if (ccsize < _MinCCAreaFrac * tilesize) { continue; }

            if (DO_BOOTSTRAP)
            {
                // Estimate bootstrap phase bias.
                float bsphase;
                BootstrapStatus_t status = estimBootstrapPhase(
                    &bsphase, &unw[bsoff], &currcc[bsoff], bsunw, bsccl, width, 
                    _NumBsLines, _MinBsPts, _BsPhaseVarThr);
    
                switch(status)
                {
                    case BootstrapSuccess:
                    {
                        // Successfully obtained bootstrap phase estimate. 
                        // Apply phase bootstrapping.

                        // Get previous connected component's label from 
                        // bootstrap overlap region. 
                        // (If there was overlap with multiple connected 
                        // components, their labels will be merged later.)
                        uint8_t bslabel = bootstrapLabel(
                            labelmap, &currcc[bsoff], bsccl, width, _NumBsLines);

                        // Apply bootstrap phase and label.
                        for (size_t i = 0; i < tilesize; ++i)
                        {
                            if (currcc[i])
                            {
                                unw[i] -= bsphase;
                                ccl[i] = bslabel;
                            }
                        }
                        break;
                    }
                    case NoBootstrap:
                    {
                        // No overlap/insufficient overlap in bootstrap region.
                        // Don't apply phase bootstrapping. Assign connected 
                        // component a new unique label.
                        uint8_t newlabel = labelmap.nextlabel();
                        for (size_t i = 0; i < tilesize; ++i)
                        {
                            if (currcc[i]) { ccl[i] = newlabel; }
                        }
                        break;
                    }
                    case BootstrapFailure:
                    {
                        // Bootstrap phase variance exceeds threshold. Restart 
                        // unwrapping with increased correlation threshold.
                        if (corrthr < _MaxCorrThr)
                        {
                            corrthr += _CorrThrInc;
                            return growGrass<DO_BOOTSTRAP>(
                                unw, ccl, currcc, bsunw, bsccl, labelmap, phase, 
                                tree, corr, corrthr, length, width);
                        }
                        else
                        {
                            throw std::runtime_error("failed to unwrap tile at max correlation threshold");
                        }
                        break;
                    }
                }
            }
            else
            {
                // Don't apply phase bootstrapping. Assign connected component 
                // a new unique label.
                uint8_t newlabel = labelmap.nextlabel();
                for (size_t i = 0; i < tilesize; ++i)
                {
                    if (currcc[i]) { ccl[i] = newlabel; }
                }
            }
        }
    }

    // Clean up unwrapped phase by deleting regions not part of any labelled 
    // connected component.
    for (size_t i = 0; i < tilesize; ++i) { if (ccl[i] == 0) { unw[i] = 0.f; } }
}

// Explicit template instantiation
template void ICU::growGrass<true>(
    float * unw, uint8_t * ccl, bool * currcc, float * bsunw, uint8_t * bsccl, 
    LabelMap & labelmap, const float * phase, const bool * tree, 
    const float * corr, float corrthr, const size_t length, const size_t width);

template void ICU::growGrass<false>(
    float * unw, uint8_t * ccl, bool * currcc, float * bsunw, uint8_t * bsccl, 
    LabelMap & labelmap, const float * phase, const bool * tree, 
    const float * corr, float corrthr, const size_t length, const size_t width);

}

