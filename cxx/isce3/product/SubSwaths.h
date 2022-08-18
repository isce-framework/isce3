//-*- C++ -*-
//-*- coding: utf-8 -*-

#pragma once

#include <string>
#include <vector>
#include <isce3/core/Matrix.h>
#include <isce3/except/Error.h>

namespace isce3 { namespace product {

/**
 * Sub-swaths metadata of a SAR dataset.
 *
 * This class holds attributes of a SAR dataset such as arrays that indicate
 * valid samples within each sub-swath. The class also provides a method
 * (`getSampleSubSwath()`) to test if a given radar
 * sample (identified by its associated range and azimuth index) belongs
 * to a certain sub-swath or not, in which case the radar sample is
 * considered invalid.
 */
class SubSwaths {

public:
    // Construct a new SubSwaths object. 
    inline SubSwaths() :
        _rlength(1),
        _rwidth(1) {};

    /**
     * Construct a new SubSwaths object.
     *
     * @param[in] length Radar grid length
     * @param[in] width  Radar grid width
     * @param[in] n      Number of sub-swaths
     */
     inline SubSwaths(const int length, const int width, const int n = 1) : 
        _rlength(length),
        _rwidth(width) {
            numSubSwaths(n);
    }

    /**
     * Construct a new SubSwaths object.
     *
     * @param[in] length Radar grid length
     * @param[in] width  Radar grid width
     * @param[in] v vector of arrays of dimensions `Lx2` (for each sub-swath)
     * indicating the indices of the first valid range sample and next sample
     * after the last valid sample for each azimuth line.
     */
     inline SubSwaths(const int length, const int width,
                      const std::vector<isce3::core::Matrix<int>> & v) :
        _rlength(length),
        _rwidth(width) {
        setValidSamplesArraysVect(v);
    }

    /** Get number of sub-swaths */
    inline int numSubSwaths() const { return _validSamplesArraysVect.size(); }
    /** Set number of sub-swaths */
    inline void numSubSwaths(const int n) {
        if (n <= 0) {
            std::string error_msg =
                    "ERROR the number of sub-swaths must be greater than zero";
            throw isce3::except::OutOfRange(ISCE_SRCINFO(), error_msg);
        }
        _validSamplesArraysVect.resize(n, isce3::core::Matrix<int>());
        validate();
    };

    /** Get valid samples for a sub-swath's array indexed from 1 (1st sub-swath)
     *
     * @param[in] n subswath index (1-based)
     * @return array of dimensions `Lx2` indicating the indices of the first
     * valid range sample and next sample after the last
     * valid range sample for each azimuth line of sub-swath `n`.
     */
    const isce3::core::Matrix<int>& getValidSamplesArray(const int n) const;

    /** Set valid samples for a sub-swath's array indexed from 1 (1st sub-swath) 
     *
     * @param[in] n subswath index (1-based)
     * @param[in] v array of dimensions `Lx2` indicating the indices of the first
     * valid range sample and next sample after the last
     * valid range sample for each azimuth line of sub-swath `n`.
     * If the index of the first valid range sample is greater or equal
     * to the index of the last valid sample, it is considered that
     * the azimuth line does not have valid samples.
     */
    void setValidSamplesArray(const int n, const isce3::core::Matrix<int>& v);

    /** Get valid samples sub-swaths vector of arrays
     *
     * @return vector of arrays of dimensions `Lx2` (for each sub-swath)
     * indicating the indices of the first valid range sample and next sample
     * after the last valid sample for each azimuth
     * line. If the index of the first valid range sample is greater or equal
     * to the index of the last valid sample, it is considered that
     * the azimuth line does not have valid samples.
    */
    const std::vector<isce3::core::Matrix<int>>& getValidSamplesArraysVect() const{
        return _validSamplesArraysVect;
    };

    /** Set valid samples sub-swaths vector of arrays
     *
     * @param[in] v vector of arrays of dimensions `Lx2` (for each sub-swath)
     * indicating the indices of the first valid range sample and next sample
     * after the last valid sample for each azimuth
     * line. If the index of the first valid range sample is greater
     * or equal to the index of the last valid sample, it is considered that
     * the azimuth line does not have valid samples.
    */
    void setValidSamplesArraysVect(const std::vector<isce3::core::Matrix<int>>& v){
        _validSamplesArraysVect = v;
        validate();
    };

    /** Test if a radar sample belongs to a sub-swath or if it is invalid.
     *
     * Returns the 1-based index of the subswath that contains the pixel
     * indexed by `azimuth_index` and `range_index`. If the pixel was not
     * a member of any subswath, returns 0.
     *
     * If the dataset does not have sub-swaths valid-samples metadata, the
     * dataset is considered to have a single sub-swath and all samples are
     * treated as belonging to that first sub-swath. If sub-swath
     * valid-samples are not provided for an existing sub-swath `s`
     * (i.e. that subswath vector is empty), all samples of that sub-swath `s`
     * will be considered valid. If more than one sub-swath has no sub-swaths
     * valid-samples information, only the first sub-swath without
     * valid-samples information will be returned. If the index of the first valid
     * range sample is greater than or equal to the index of the last valid sample,
     * it is considered that the azimuth line does not have valid samples.
     *
     * @param[in]  azimuth_index  Azimuth index
     * @param[in]  range_index    Range index
     * @return     The number of the first sub-swath that contains the radar
     * sample, or zero, if the radar sample is invalid (i.e., not contained
     * in any sub-swaths).
     */
    int getSampleSubSwath(const int azimuth_index, const int range_index) const;

    /** Test if a radar sample belongs to a sub-swath or if it is invalid.
     *
     * Returns the 1-based index of the subswath that contains the pixel
     * indexed by `azimuth_index` and `range_index`. If the pixel was not
     * a member of any subswath, returns 0.
     *
     * If the dataset does not have sub-swaths valid-samples metadata, the
     * dataset is considered to have a single sub-swath and all samples are
     * treated as belonging to that first sub-swath. If sub-swath
     * valid-samples are not provided for an existing sub-swath `s`
     * (i.e. that subswath vector is empty), all samples of that sub-swath `s`
     * will be considered valid. If more than one sub-swath has no sub-swaths
     * valid-samples information, only the first sub-swath without
     * valid-samples information will be returned. If the index of the first valid
     * range sample is greater than or equal to the index of the last valid sample,
     * it is considered that the azimuth line does not have valid samples.
     *
     * @param[in]  azimuth_index  Azimuth index
     * @param[in]  range_index    Range index
     * @return     The number of the first sub-swath that contains the radar
     * sample, or zero, if the radar sample is invalid (i.e., not contained
     * in any sub-swaths).
     */
    int operator()(const int azimuth_index, const int range_index) const {
        return getSampleSubSwath(azimuth_index, range_index);
    };

    /** Get the radar grid length */
    inline int length() const { return _rlength; }

    /** Set the radar grid length */
    inline void length(const int & t) {
        _rlength = t;
        validate();
    }

    /** Get the radar grid width */
    inline int width() const { return _rwidth; }

    /** Set the radar grid width */
    inline void width(const int & t) { 
        _rwidth = t;
        validate();
    }

private:

    /** Radar grid length */
    int _rlength;

    /** Radar grid width */
    int _rwidth;

    /** Vector of arrays representing the valid samples for each
        sub-swath */
    std::vector<isce3::core::Matrix<int>> _validSamplesArraysVect;

    /** Validate SubSwaths` parameters */
    inline void validate();
};

// Validation of SubSwaths attributes
void
isce3::product::SubSwaths::validate() {

    std::string error_str = "";

    if (length() <= 0)
    {
        error_str += "Number of azimuth lines must be positive. \n";
    }

    if (width() <= 0)
    {
        error_str += "Number of range samples must be positive. \n";
    }

    for (int s = 0; s < _validSamplesArraysVect.size(); ++s) {
        if (_validSamplesArraysVect[s].size() == 0) {
            continue;
        }
        if (_validSamplesArraysVect[s].width() != 2) {
            error_str += "The valid samples array of sub-swath ";
            error_str += std::to_string(s + 1);
            error_str += " does not have two columns. The";
            error_str += " columns should represent the indices";
            error_str += " of the first valid range sample and next";
            error_str += " sample after the last valid sample";
            error_str += " for each azimuth line, respectively.\n";
            continue;
        }
        if (_validSamplesArraysVect[s].length() != length()) {
            error_str += "The valid samples array of sub-swath ";
            error_str += std::to_string(s + 1);
            error_str += " has ";
            error_str += std::to_string(_validSamplesArraysVect[s].length());
            error_str += " lines whereas the number of azimuth lines is ";
            error_str += std::to_string(length());
            error_str += ".\n";
        }
        bool flag_informed_error_negative = false;
        bool flag_informed_error_width = false;
        for (int i = 0; i < length(); ++i) {

            for (int j = 0; j < 2; ++j) {

                /** Check for invalid values in `_validSamplesArraysVect`.
                */
                if (_validSamplesArraysVect[s](i, j) < 0 and
                        !flag_informed_error_negative) {
                    error_str += "The valid samples array of sub-swath ";
                    error_str += std::to_string(s + 1);
                    error_str += " has negative indices. For example,";
                    error_str += " the array element at azimuth line ";
                    error_str += std::to_string(i);
                    error_str += " and column ";
                    error_str += std::to_string(j);
                    error_str += " has value ";
                    error_str += std::to_string(_validSamplesArraysVect[s](i, j));
                    error_str += ".\n";
                }
                if (_validSamplesArraysVect[s](i, j) > width() and
                        !flag_informed_error_width) {
                    error_str += "The valid samples array of sub-swath ";
                    error_str += std::to_string(s + 1);
                    error_str += " has invalid range indices.";
                    error_str += " The array element at azimuth line ";
                    error_str += std::to_string(i);
                    error_str += " and column ";
                    error_str += std::to_string(j);
                    error_str += " has value ";
                    error_str += std::to_string(_validSamplesArraysVect[s](i, j));
                    error_str += " which is invalid for the sub-swath with ";
                    error_str += std::to_string(width());
                    error_str += " range samples.\n";
                    flag_informed_error_width = true;
                }

            }

            // If both error messages were shown to the user, exit loop
            if (flag_informed_error_negative and flag_informed_error_width) {
                break;
            }
        }
    }

    if (! error_str.empty())
    {
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), error_str);
    }
}


}}
