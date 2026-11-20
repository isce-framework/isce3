#pragma once

#include <memory>
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <isce3/focus/Backproject.h>
#include <isce3/geometry/detail/Rdr2Geo.h>
#include <isce3/geometry/detail/Geo2Rdr.h>

void addbinding(pybind11::class_<isce3::focus::PolarGrid>& pyPolarGrid);
void addbinding_backproject(pybind11::module& m);

isce3::geometry::detail::Rdr2GeoBracketParams
parse_rdr2geo_params(const pybind11::dict& params);

isce3::geometry::detail::Geo2RdrBracketParams
parse_geo2rdr_params(const pybind11::dict& params);

/**
 * Transfer ownership of a unique_ptr<T[]> buffer to a numpy array without
 * copying.  The array takes ownership via a capsule whose destructor deletes[]
 * the buffer when the last reference dies.
 *
 * \param ptr       Unique pointer to transfer (will be released)
 * \param shape     Array shape (C-order / row-major is assumed)
 * \return          numpy array wrapping the transferred buffer
 */
template<class T>
pybind11::array_t<T> move_to_numpy(
    std::unique_ptr<T[]> ptr,
    std::vector<pybind11::ssize_t> shape)
{
    // Compute C-order strides.
    std::vector<pybind11::ssize_t> strides(shape.size());
    pybind11::ssize_t stride = sizeof(T);
    for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }

    T* data = ptr.release();

    pybind11::capsule owner(data, [](void *p) {
        delete[] static_cast<T*>(p);
    });

    return pybind11::array_t<T>(shape, strides, data, owner);
}
