#include "SubSwaths.h"

#include <iostream>

#include <pyre/journal.h>

#include <isce3/except/Error.h>

namespace isce3 { namespace product {

const isce3::core::Matrix<int>& isce3::product::SubSwaths::getValidSamplesArray(
        const int n) const
{
    if (n <= 0) {
        std::string error_msg =
                "ERROR the sub-swath number must be greater than zero";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);
    }
    if (n > _validSamplesArraysVect.size()) {
        std::string error_msg =
                "ERROR cannot read valid samples array for sub-swath ";
        error_msg += std::to_string(n) + ". Dataset has only ";
        error_msg +=
                std::to_string(_validSamplesArraysVect.size()) + " sub-swaths.";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);
    }
    return _validSamplesArraysVect[n - 1];
}

void isce3::product::SubSwaths::setValidSamplesArray(
        const int n, const isce3::core::Matrix<int>& v)
{
    if (n <= 0) {
        std::string error_msg =
            "ERROR cannot assign valid-samples array to sub-swath ";
        error_msg += std::to_string(n) + ". The sub-swath number must be";
        error_msg += " greater than zero.";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);
    }
    if (n <= _validSamplesArraysVect.size()) {
        _validSamplesArraysVect.erase(_validSamplesArraysVect.begin() + n - 1);
    }
    else if (n > _validSamplesArraysVect.size()) {
        if (n > _validSamplesArraysVect.size() + 1) {
            pyre::journal::warning_t warning("isce.product.SubSwaths");
            warning << "WARNING since this dataset contains "
                    << _validSamplesArraysVect.size()
                    << " sub-swaths, adding a valid-samples array"
                    << " to sub-swath " << n << " creates"
                    << " empty sub-swaths." << pyre::journal::endl;
        }
        _validSamplesArraysVect.resize(n - 1, isce3::core::Matrix<int>());
    }
    _validSamplesArraysVect.emplace(_validSamplesArraysVect.begin() + n - 1, v);
    validate();
}

int isce3::product::SubSwaths::getSampleSubSwath(
        const int azimuth_index, const int range_index) const
{

    // If invalid range or azimuth indices, return 0 (invalid)
    if (range_index < 0 || range_index >= _rwidth ||
            azimuth_index < 0 || azimuth_index >= _rlength) {
        return 0;
    }

    /* If the dataset does not have sub-swaths information,
    consider samples valid and belonging to the first sub-swath */
    if (_validSamplesArraysVect.size() == 0) {
        return 1;
    }

    for (int s = 1; s <= _validSamplesArraysVect.size(); ++s) {
        /* If any subswath was empty, then we consider all samples in the
           swath as valid samples of that subswath */
        if (_validSamplesArraysVect[s - 1].size() == 0) {
            return s;
        }

        // Range index start and end (exclusive) with half-open interval
        const int range_index_start =
                _validSamplesArraysVect[s - 1](azimuth_index, 0);
        const int range_index_end =
                _validSamplesArraysVect[s - 1](azimuth_index, 1);

        // if radar sample is valid for sub-swath "s", return "s"
        if (range_index >= range_index_start &&
                range_index < range_index_end) {
            return s;
        }
    }
    return 0;
}

}}
