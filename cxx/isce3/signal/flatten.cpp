#include "flatten.h"

#include <isce3/except/Error.h>

void isce3::signal::flatten(isce3::core::EArray2D<std::complex<float>>& ifgram,
                            const isce3::core::EArray2D<double>& range_offset,
                            double range_spacing, double wavelength)
{
    if (range_offset.rows() != ifgram.rows()) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Number of rows in range offsets must be equal "
                                "to the number of rows in the interferogram.");
    }
    if (range_offset.cols() != ifgram.cols()) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Number of columns in range offsets must be equal to the "
                "number of columns in the interferogram .");
    }
    if (range_spacing <= 0.0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Slant range spacing must be a positive number.");
    }
    if (wavelength <= 0.0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Radar wavelength must be a positive number.");
    }
    isce3::core::EArray2D<std::complex<float>> geometry_ifg(ifgram.rows(),
                                                            ifgram.cols());
    auto range_offset_phase =
            4.0 * M_PI * range_spacing * range_offset / wavelength;

    // store phase in complex container
    geometry_ifg.real() = range_offset_phase.cos().cast<float>();
    geometry_ifg.imag() = range_offset_phase.sin().cast<float>();

    ifgram *= geometry_ifg.conjugate();
}
