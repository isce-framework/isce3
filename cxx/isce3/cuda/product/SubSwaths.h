#pragma once

#include <vector>
#include <thrust/device_vector.h>

#include <isce3/core/Common.h>
#include <isce3/product/SubSwaths.h>


namespace isce3::cuda::product {

/** Base class for GPU subswath.
 */
class BaseSubSwaths {
protected:
    // Length of radar grid
    size_t _length = 0;

    // Width of radar grid
    size_t _width = 0;

    // Number of subswaths
    unsigned int _n_subswaths = 0;

public:
    /* Default constructor has no subswaths and results in no masking */
    CUDA_HOST BaseSubSwaths() {}

    /**
     * @brief Construct with an existing CPU subswath object.
     * @param[in] cpu_subswaths The CPU subswath object.
     * @note Error checking is absent; rely on error checking performed in the construction
     * of the CPU subswath object.
     */
    CUDA_HOST BaseSubSwaths(const isce3::product::SubSwaths& cpu_subswaths):
        _length(cpu_subswaths.length()),
        _width(cpu_subswaths.width())
    {
        // CPU subswath defaults the number of subswaths to 1 even if the RSLC
        // has no subswaths.
        // Need to check if samples array is empty to determine if number of
        // subswaths is zero.
        const auto sz_1st_subswath = cpu_subswaths.getValidSamplesArray(1).size();
        if (cpu_subswaths.numSubSwaths() == 1 and sz_1st_subswath == 0)
            _n_subswaths = 0;
        else
            _n_subswaths = cpu_subswaths.numSubSwaths();
    }

    /**
     * @brief Construct a BaseSubSwaths object with specified parameters.
     * @param[in] length The length of the radar grid.
     * @param[in] width The width of the radar grid.
     * @param[in] n_subswaths The number of subswaths.
     */
    CUDA_HOST BaseSubSwaths(const size_t length, const size_t width,
            unsigned int n_subswaths):
        _length(length),
        _width(width),
        _n_subswaths(n_subswaths) {}

    /**
     * @brief Get the length of the radar grid.
     */
    CUDA_HOSTDEV size_t length() const { return _length; }

    /**
     * @brief Get the width of the radar grid.
     */
    CUDA_HOSTDEV size_t width() const { return _width; }

    /**
     * @brief Get the number of subswaths.
     */
    CUDA_HOSTDEV unsigned int n_subswaths() const { return _n_subswaths; }
};


/** Owner subswaths class for use in GPU geocode. Only created from CPU
 *  SubSwaths object. Each CPU subswath matrix is stored as a device vector and
 *  owned by this class so it can be persisted.
 */
class OwnerSubSwaths : public BaseSubSwaths {
private:
    /* device_vector holding all subswath starts.
     * Has dimensions 1 x (number of subswaths * slc length) */
    thrust::device_vector<int> _valid_start_samples;

    /* device_vector holding all subswath stops.
     * Has dimensions 1 x (number of subswaths * slc length) */
    thrust::device_vector<int> _valid_stop_samples;

public:
    /**
     * @brief Default constructor initializing no subswaths so no masking occurs.
     */
    CUDA_HOST OwnerSubSwaths(): BaseSubSwaths() {}

    /**
     * @brief Constructor that copies data from a CPU SubSwaths object.
     * @param[in] cpu_subswaths The CPU SubSwaths object.
     */
    CUDA_HOST OwnerSubSwaths(const isce3::product::SubSwaths& cpu_subswaths);

    /**
     * @brief Get pointer to device_vector of sample starts.
     */
    CUDA_HOST int* ptr_to_valid_start()
    {
        return thrust::raw_pointer_cast(_valid_start_samples.data());
    }

    /**
     * @brief Get const pointer to device_vector of sample starts.
     */
    CUDA_HOST const int* ptr_to_valid_start() const
    {
        return thrust::raw_pointer_cast(_valid_start_samples.data());
    }

    /**
     * @brief Get pointer to device_vector of sample stops.
     */
    CUDA_HOST int* ptr_to_valid_stop()
    {
        return thrust::raw_pointer_cast(_valid_stop_samples.data());
    }

    /**
     * @brief Get const pointer to device_vector of sample stops.
     */
    CUDA_HOST const int* ptr_to_valid_stop() const
    {
        return thrust::raw_pointer_cast(_valid_stop_samples.data());
    }
};


/** View subswaths class. Only created from OwnerSubSwaths objects.Each
 *  subswath device vector in the OwnerSubSwaths is viewable as a pointer to
 *  the device vector's data on the GPU.
 */
class ViewSubSwaths : public BaseSubSwaths {
private:
    /**
     * @brief Pointers to device vectors' data buffers, where each pointer provides
     *        a view into the valid samples data for a specific subswath. These views
     *        allow direct access to the valid samples information stored in the device
     *        vectors, facilitating efficient GPU processing without the need for data
     *        transfers between CPU and GPU.
     */
    int* _valid_start_view = nullptr;
    int* _valid_stop_view = nullptr;

public:
    /**
     * @brief Default constructor uninitialized with an owner needed for test
     *        harness. Not to be used by ordinary code.
     */
    CUDA_HOST ViewSubSwaths(): BaseSubSwaths() {}

    /**
     * @brief Constructor to create ViewSubSwaths from an OwnerSubSwaths object.
     * @param[in] owner_subswaths The OwnerSubSwaths object to create views from.
     */
    /* if const OwnerSubSwaths& as paramter then
     * error: the object has type qualifiers that are not compatible with the member function "isce3::cuda::product::OwnerSubSwaths::ptr_to_valid_stop"
     */
    CUDA_HOST ViewSubSwaths(OwnerSubSwaths& owner_subswaths);

    /**
     * @brief Check if a specific index is contained within the subswaths.
     * @param[in] index_aztime The azimuth time index to check.
     * @param[in] index_srange The slant range index to check.
     * @return True if the index is contained within any subswath, false otherwise. If no subswaths, automatically return true.
     */
    CUDA_DEV bool contains(const int index_aztime, const int index_srange) const
    {
        // _n_subswaths == 0 indicates no SubSwaths i.e. no masking so return
        // true
        if (_n_subswaths == 0)
            return true;

        // Check if az and srg in bounds of radar grid, return false if out of
        // bounds
        if (index_aztime < 0 || index_aztime >= _length || index_srange < 0 ||
                index_srange >= _width)
            return false;

        for (unsigned int i_subswath = 0; i_subswath < _n_subswaths; ++i_subswath)
        {
            // Compute pointer arithmetic offset of each subswath
            const auto swath_offset = i_subswath * _length;

            // Get start of current subswath block
            const auto subswath_rg_start = *(_valid_start_view + swath_offset + index_aztime);

            // Get stop of current subswath block
            const auto subswath_rg_stop = *(_valid_stop_view + swath_offset + index_aztime);

            if (index_srange >= subswath_rg_start && index_srange < subswath_rg_stop)
                return true;
        }

        return false;
    }
};

} // end namespace isce3::cuda::product
