#include "RadarGridParameters.h"

#include "Product.h"

isce3::product::RadarGridParameters::
RadarGridParameters(const Product & product, char frequency) :
    RadarGridParameters(product.swath(frequency), product.lookSide())
{
    validate();
}

isce3::product::RadarGridParameters::
RadarGridParameters(const Swath & swath, isce3::core::LookSide lookSide) :
    _lookSide(lookSide),
    _sensingStart(swath.zeroDopplerTime()[0]),
    _wavelength(swath.processedWavelength()),
    _prf(swath.nominalAcquisitionPRF()),
    _startingRange(swath.slantRange()[0]),
    _rangePixelSpacing(swath.rangePixelSpacing()),
    _rlength(swath.lines()),
    _rwidth(swath.samples()),
    _refEpoch(swath.refEpoch())
{
    validate();
}
