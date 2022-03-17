#include "RadarGridParameters.h"

#include "RadarGridProduct.h"

isce3::product::RadarGridParameters::
RadarGridParameters(const RadarGridProduct & product, char frequency) :
    RadarGridParameters(product.swath(frequency), product.lookSide())
{
    validate();
}

isce3::product::RadarGridParameters::
RadarGridParameters(const Swath & swath, isce3::core::LookSide lookSide) :
    _lookSide(lookSide),
    _sensingStart(swath.zeroDopplerTime()[0]),
    _wavelength(swath.processedWavelength()),
    _prf(1.0 / swath.zeroDopplerTimeSpacing()),
    _startingRange(swath.slantRange()[0]),
    _rangePixelSpacing(swath.rangePixelSpacing()),
    _rlength(swath.lines()),
    _rwidth(swath.samples()),
    _refEpoch(swath.refEpoch())
{
    validate();
}
