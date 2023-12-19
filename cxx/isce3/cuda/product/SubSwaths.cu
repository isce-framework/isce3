#include "SubSwaths.h"

#include <isce3/cuda/except/Error.h>

namespace isce3::cuda::product {

__host__
OwnerSubSwaths::OwnerSubSwaths(const isce3::product::SubSwaths& cpu_subswaths):
    BaseSubSwaths(cpu_subswaths)
{
    if (_n_subswaths > 0)
    {
        const auto n_samples_all_subswaths  = _n_subswaths * _length;

        // Create temp host of above to faciltiate copying.
        // A vector of indices per subswath will not work so starts and stops
        // of all subswaths will be concatenated together.
        std::vector<int> host_valid_start_samples(n_samples_all_subswaths);
        std::vector<int> host_valid_stop_samples(n_samples_all_subswaths);

        // Iterate through host subswaths by element and copy to temp vector.
        // Can not do a cudaMemcpy because CPU array is column major and
        // that operation would not create contiguous block of subswath starts
        // and stops.
        for (unsigned int i = 0; i < _n_subswaths; ++i)
        {
            // Get reference to source array and number of elements in it
            // + 1 because isce3::product::SubSwath starts index at 1
            const auto cpu_valid_arr = cpu_subswaths.getValidSamplesArray(i + 1);

            // Compute pointer arithmetic offset.
            const auto subswath_offset = i * _length;

            // Copy from host to temp starts/stops.
            for (size_t i_cpu = 0; i_cpu < _length; ++i_cpu)
            {
                host_valid_start_samples[i_cpu + subswath_offset] =
                    cpu_valid_arr(i_cpu, 0);
                host_valid_stop_samples[i_cpu + subswath_offset] =
                    cpu_valid_arr(i_cpu, 1);
            }
        }

        // Copy to device.
        _valid_start_samples = host_valid_start_samples;
        _valid_stop_samples = host_valid_stop_samples;
    }
}


__host__
ViewSubSwaths::ViewSubSwaths(OwnerSubSwaths& owner_subswaths):
    BaseSubSwaths(owner_subswaths.length(),
                  owner_subswaths.width(),
                  owner_subswaths.n_subswaths())
{
    if (_n_subswaths > 0)
    {
        _valid_start_view = owner_subswaths.ptr_to_valid_start();
        _valid_stop_view = owner_subswaths.ptr_to_valid_stop();
    }
}

}
