/*
 * @file cuNormalizer.cu
 * @brief processors to normalize the correlation surface
 *
 */

#include "cuCorrNormalizer.h"
#include "cuAmpcorUtil.h"

namespace isce3::matchtemplate::pycuampcor {

cuNormalizer::cuNormalizer(int secondaryNX, int secondaryNY, int count)
{
    // Always use SAT method for CPU, since
    // blocksize-dependent optimizations are GPU-specific
    processor = new cuNormalizeSAT(secondaryNX, secondaryNY, count);
}

cuNormalizer::~cuNormalizer()
{
    delete processor;
}

void cuNormalizer::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary)
{
    processor->execute(correlation, reference, secondary);
}

/**
 *
 *
 **/

cuNormalizeSAT::cuNormalizeSAT(int secondaryNX, int secondaryNY, int count)
{
    // allocate the work array
    // reference sum square
    referenceSum2 = new cuArrays<float>(1, 1, count);
    referenceSum2->allocate();

    // secondary sum and sum square
    secondarySAT = new cuArrays<float>(secondaryNX, secondaryNY, count);
    secondarySAT->allocate();
    secondarySAT2 = new cuArrays<float>(secondaryNX, secondaryNY, count);
    secondarySAT2->allocate();
};

cuNormalizeSAT::~cuNormalizeSAT()
{
    delete referenceSum2;
    delete secondarySAT;
    delete secondarySAT2;
}

void cuNormalizeSAT::execute(cuArrays<float> *correlation,
    cuArrays<float> *reference, cuArrays<float> *secondary)
{
    cuCorrNormalizeSAT(correlation, reference, secondary,
        referenceSum2, secondarySAT, secondarySAT2);
}

} // namespace
