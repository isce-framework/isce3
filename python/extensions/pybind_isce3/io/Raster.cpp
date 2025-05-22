#include "Raster.h"

#include <algorithm>
#include <complex>
#include <gdal_priv.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include <isce3/except/Error.h>
#include <isce3/io/IH5Dataset.h>

namespace py = pybind11;

using isce3::io::Raster;

using SlicePair = std::pair<py::slice, py::slice>;

/**
 * Check that `band` is a valid band index in `raster`.
 *
 * @param[in] raster
 *     The input raster dataset.
 * @param[in] band
 *     The (1-based) index of the raster band.
 *
 * @throws isce3::except::OutOfRange
 *     If the raster dataset did not contain the specified band.
 */
void ensureValidBand(const Raster& raster, int band)
{
    const auto num_bands = raster.numBands();
    if ((band < 1) || (band > num_bands)) {
        const auto errmsg = "band " + std::to_string(band) +
                            " is out of bounds for a raster dataset with " +
                            std::to_string(num_bands) + " bands";
        throw isce3::except::OutOfRange(ISCE_SRCINFO(), errmsg);
    }
}

/**
 * Check that the raster dataset contains a single raster band.
 *
 * @param[in] raster
 *     The input raster dataset.
 *
 * @throws isce3::except::InvalidArgument
 *     If the dataset contained more than one raster band.
 */
void ensureSingleBand(const Raster& raster)
{
    const auto n = raster.numBands();
    if (n != 1) {
        const auto errmsg = "only single-band raster datasets are supported, "
                            "got numBands=" +
                            std::to_string(n);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
}

/**
 * Get the datatype of a raster as a `numpy.dtype` object.
 *
 * @param[in] raster
 *     The input raster dataset.
 * @param[in] band
 *     The (1-based) index of the raster band to get the datatype of. Defaults
 *     to 1.
 *
 * @returns
 *     The raster datatype.
 */
py::dtype getRasterDType(const Raster& raster, std::size_t band = 1)
{
    const auto dtype = raster.dtype(band);
    switch (dtype) {
        // clang-format off
        case GDT_Byte:     return py::dtype::of<unsigned char>();
#if GDAL_VERSION_NUM >= 3070000 // (GDAL>=3.7)
        case GDT_Int8:     return py::dtype::of<std::int8_t>();
#endif
        case GDT_UInt16:   return py::dtype::of<std::uint16_t>();
        case GDT_Int16:    return py::dtype::of<std::int16_t>();
        case GDT_UInt32:   return py::dtype::of<std::uint32_t>();
        case GDT_Int32:    return py::dtype::of<std::int32_t>();
#if GDAL_VERSION_NUM >= 3050000 // (GDAL>=3.5)
        case GDT_UInt64:   return py::dtype::of<std::uint64_t>();
        case GDT_Int64:    return py::dtype::of<std::int64_t>();
#endif
        case GDT_Float32:  return py::dtype::of<float>();
        case GDT_Float64:  return py::dtype::of<double>();
        case GDT_CFloat32: return py::dtype::of<std::complex<float>>();
        case GDT_CFloat64: return py::dtype::of<std::complex<double>>();
        default: break;
        // clang-format on
    }

    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "unexpected GDAL datatype: " + std::to_string(dtype));
}

/**
 * Get the start index and size of a contiguous slice.
 *
 * @param[in] slice
 *     The input slice object. Must be a contiguous slice (with step == 1).
 * @param[in] length
 *     The length of the sequence to be sliced.
 *
 * @returns
 *     The start index of the slice.
 * @returns
 *     The length of the slice.
 */
std::pair<std::size_t, std::size_t> getContiguousSliceStartAndSize(
        const py::slice& slice, std::size_t length)
{
    // Get the start & stop indices, step size, and slice length. Out-of-bounds
    // indices are clipped according to normal slicing rules.
    std::size_t start, stop, step, slicelength;
    if (!slice.compute(length, &start, &stop, &step, &slicelength)) {
        throw py::error_already_set();
    }

    // Check that the slice is contiguous.
    if (step != 1) {
        const auto errmsg = "only contiguous slices are supported, got step=" +
                            std::to_string(step);
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }

    return std::pair(start, slicelength);
}

/**
 * Read a block of data from a raster as the specified type.
 *
 * @tparam T
 *     The element type of the output array. Values will be converted to this
 *     type if different from the raster storage type. Must be a valid datatype
 *     supported by GDAL.
 *
 * @param[in] raster
 *     The raster to read from.
 * @param[in] key
 *     The row and column indices of the block to read from the raster. Normal
 *     Python slicing rules apply.
 * @param[in] band
 *     The (1-based) index of the raster band to read from. Defaults to 1.
 *
 * @returns
 *     A 2-D NumPy array with the desired datatype containing values read from
 *     the raster.
 */
template<class T>
py::array_t<T> getRasterBlockAsType(
        Raster& raster, const SlicePair& key, int band = 1)
{
    ensureValidBand(raster, band);

    // Get the start index and length of each slice. Only contiguous
    // (non-strided) slices are supported currently.
    const auto [row_start, num_rows] =
            getContiguousSliceStartAndSize(key.first, raster.length());
    const auto [col_start, num_cols] =
            getContiguousSliceStartAndSize(key.second, raster.width());

    // Load the corresponding block of data from the raster.
    auto arr = py::array_t<T>({num_rows, num_cols});
    raster.getBlock(
            arr.mutable_data(), col_start, row_start, num_cols, num_rows, band);

    return arr;
}

/**
 * Read a block of data from a raster.
 *
 * @param[in] raster
 *     The raster to read from.
 * @param[in] key
 *     The row and column indices of the block to read from the raster. Normal
 *     Python slicing rules apply.
 * @param[in] band
 *     The (1-based) index of the raster band to read from. Defaults to 1.
 *
 * @returns
 *     A 2-D NumPy array with the same datatype as the raster containing the
 *     values read from the raster.
 */
py::array getRasterBlock(Raster& raster, const SlicePair& key, int band = 1)
{
    switch (raster.dtype()) {
        // clang-format off
        case GDT_Byte:     return getRasterBlockAsType<unsigned char>(raster, key, band);
#if GDAL_VERSION_NUM >= 3070000 // (GDAL>=3.7)
        case GDT_Int8:     return getRasterBlockAsType<std::int8_t>(raster, key, band);
#endif
        case GDT_UInt16:   return getRasterBlockAsType<std::uint16_t>(raster, key, band);
        case GDT_Int16:    return getRasterBlockAsType<std::int16_t>(raster, key, band);
        case GDT_UInt32:   return getRasterBlockAsType<std::uint32_t>(raster, key, band);
        case GDT_Int32:    return getRasterBlockAsType<std::int32_t>(raster, key, band);
#if GDAL_VERSION_NUM >= 3050000 // (GDAL>=3.5)
        case GDT_UInt64:   return getRasterBlockAsType<std::uint64_t>(raster, key, band);
        case GDT_Int64:    return getRasterBlockAsType<std::int64_t>(raster, key, band);
#endif
        case GDT_Float32:  return getRasterBlockAsType<float>(raster, key, band);
        case GDT_Float64:  return getRasterBlockAsType<double>(raster, key, band);
        case GDT_CFloat32: return getRasterBlockAsType<std::complex<float>>(raster, key, band);
        case GDT_CFloat64: return getRasterBlockAsType<std::complex<double>>(raster, key, band);
        default: break;
        // clang-format on
    }

    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "unable to access raster data: unsupported GDAL datatype");
}

/** Get the shape of a NumPy array as a Python tuple. */
inline py::tuple getShapeTuple(const py::array& arr)
{
    return py::tuple(py::cast(arr.request().shape));
}

/**
 * Check that the two input shape tuples are equivalent.
 *
 * @param[in] shape1
 *     The first input shape tuple.
 * @param[in] shape2
 *     The second input shape tuple.
 *
 * @throws isce3::except::InvalidArgument
 *     If the two shapes are different.
 */
void ensureMatchingShapes(const py::tuple& shape1, const py::tuple& shape2)
{
    // Check whether both shapes have the same size and are elementwise equal.
    // Raise an exception otherwise.
    auto pred = [](const auto& a, const auto& b) {
        return py::cast<std::size_t>(a) == py::cast<std::size_t>(b);
    };
    const auto all_equal = std::equal(std::begin(shape1), std::end(shape1),
            std::begin(shape2), std::end(shape2), pred);
    if (!all_equal) {
        const auto errmsg = std::string(
                py::str("could not broadcast array from shape {} into shape {}")
                        .format(shape1, shape2));
        throw isce3::except::InvalidArgument(ISCE_SRCINFO(), errmsg);
    }
}

/**
 * Write a block of data to a raster.
 *
 * @param[in] raster
 *     The raster to write to.
 * @param[in] key
 *     The row and column indices of the block in the raster to be overwritten.
 *     Normal Python slicing rules apply.
 * @param[in] value
 *     The data to write to the raster.
 * @param[in] band
 *     The (1-based) index of the raster band to write to. Defaults to 1.
 */
template<class T>
void setRasterBlockFromArray(Raster& raster, const SlicePair& key,
        py::array_t<T, py::array::c_style | py::array::forcecast> value,
        int band = 1)
{
    ensureValidBand(raster, band);

    // Get the start index and length of each slice. Only contiguous
    // (non-strided) slices are supported currently.
    const auto [row_start, num_rows] =
            getContiguousSliceStartAndSize(key.first, raster.length());
    const auto [col_start, num_cols] =
            getContiguousSliceStartAndSize(key.second, raster.width());

    // Check that the array shape matches the slice shape.
    const auto key_shape = py::make_tuple(num_rows, num_cols);
    const auto val_shape = getShapeTuple(value);
    ensureMatchingShapes(val_shape, key_shape);

    // Write data to the raster.
    // XXX We need to use `mutable_data()` here because `Raster` can't read from
    // const pointers.
    auto ptr = value.mutable_data();
    raster.setBlock(ptr, col_start, row_start, num_cols, num_rows, band);
}

/**
 * Write a block of data to a raster as the specified type.
 *
 * Given an input buffer object, this function converts the buffer to a
 * C-contiguous NumPy array with element type `T`, then writes the array
 * contents to a subblock of the raster. The intermediate conversion to a NumPy
 * array is convenient because NumPy is able to handle strided buffers and/or
 * datatypes with non-native endianness, which otherwise would be tedious to
 * implement. In general, `T` may be any GDAL-compatible type, but typically it
 * should be the same type as the raster dataset contents (because otherwise
 * GDAL will perform a second conversion step when writing to the raster).
 *
 * @tparam T
 *     The element type that the contents of `value` will be cast to. Must be a
 *     valid datatype supported by GDAL.
 *
 * @param[in] raster
 *     The raster to write to.
 * @param[in] key
 *     The row and column indices of the block in the raster to be overwritten.
 *     Normal Python slicing rules apply.
 * @param[in] value
 *     The data to write to the raster.
 * @param[in] band
 *     The (1-based) index of the raster band to write to. Defaults to 1.
 */
template<class T>
void setRasterBlockAsType(
        Raster& raster, const SlicePair& key, py::buffer value, int band = 1)
{
    constexpr auto flags = py::array::c_style | py::array::forcecast;
    auto arr = py::cast<py::array_t<T, flags>>(value);
    setRasterBlockFromArray(raster, key, arr, band);
}

/**
 * Write a block of data to a raster.
 *
 * @param[in] raster
 *     The raster to write to.
 * @param[in] key
 *     The row and column indices of the block in the raster to be overwritten.
 *     Normal Python slicing rules apply.
 * @param[in] value
 *     The data to write to the raster.
 * @param[in] band
 *     The (1-based) index of the raster band to write to. Defaults to 1.
 */
void setRasterBlock(
        Raster& raster, const SlicePair& key, py::buffer value, int band = 1)
{
    switch (raster.dtype()) {
        // clang-format off
        case GDT_Byte:     return setRasterBlockAsType<unsigned char>(raster, key, value, band);
#if GDAL_VERSION_NUM >= 3070000 // (GDAL>=3.7)
        case GDT_Int8:     return setRasterBlockAsType<std::int8_t>(raster, key, value, band);
#endif
        case GDT_UInt16:   return setRasterBlockAsType<std::uint16_t>(raster, key, value, band);
        case GDT_Int16:    return setRasterBlockAsType<std::int16_t>(raster, key, value, band);
        case GDT_UInt32:   return setRasterBlockAsType<std::uint32_t>(raster, key, value, band);
        case GDT_Int32:    return setRasterBlockAsType<std::int32_t>(raster, key, value, band);
#if GDAL_VERSION_NUM >= 3050000 // (GDAL>=3.5)
        case GDT_UInt64:   return setRasterBlockAsType<std::uint64_t>(raster, key, value, band);
        case GDT_Int64:    return setRasterBlockAsType<std::int64_t>(raster, key, value, band);
#endif
        case GDT_Float32:  return setRasterBlockAsType<float>(raster, key, value, band);
        case GDT_Float64:  return setRasterBlockAsType<double>(raster, key, value, band);
        case GDT_CFloat32: return setRasterBlockAsType<std::complex<float>>(raster, key, value, band);
        case GDT_CFloat64: return setRasterBlockAsType<std::complex<double>>(raster, key, value, band);
        default: break;
        // clang-format on
    }

    throw isce3::except::RuntimeError(ISCE_SRCINFO(),
            "unable to access raster data: unsupported GDAL datatype");
}


void addbinding(py::class_<Raster> & pyRaster)
{
    pyRaster
        // update and read-only (default) mode
        .def(py::init([](const std::string & path, bool update)
            {
                if (update) {
                    // register IH5
                    if (path.rfind("IH5:::", 0) == 0) {
                        isce3::io::GDALRegister_IH5();
                    }
                    return std::make_unique<Raster>(path, GA_Update);
                } else {
                    return std::make_unique<Raster>(path);
                }
            }),
            "Open raster in update or read-only (default) mode",
            py::arg("path"),
            py::arg("update")=false)
        // dataset constructor
        .def(py::init([](const std::string & path, int width, int length, int num_bands,
                        int dtype, const std::string driver_name)
            {
                auto gd_dtype = static_cast<GDALDataType>(dtype);
                return std::make_unique<Raster>(path, width, length, num_bands,
                        gd_dtype, driver_name);
            }),
            "Create a raster dataset",
            py::arg("path"),
            py::arg("width"),
            py::arg("length"),
            py::arg("num_bands"),
            py::arg("dtype"),
            py::arg("driver_name"))
        // multiband constructor
        .def(py::init([](const std::string & path, std::vector<Raster> raster_list)
            {
                return std::make_unique<Raster>(path, raster_list);
            }),
            "Create a VRT raster dataset from list of rasters",
            py::arg("path"),
            py::arg("raster_list"))
        .def("close_dataset", [](Raster & self)
            {
                if (self.dataset_owner()) {
                    auto gdal_ds = self.dataset();
                    delete gdal_ds;
                }
                self.dataset(nullptr);
            },
            R"(
            Close the dataset.

            Decrements the reference count of the underlying `GDALDataset`, which,
            if this was the last open instance, causes the dataset to be closed
            and any cached changes to be flushed to disk.

            This invalidates the `Raster` instance -- it cannot be used after closing
            the underlying dataset.
            )")
        .def(py::init([](std::uintptr_t py_ds_ptr)
            {
                auto gdal_ds = reinterpret_cast<GDALDataset*>(py_ds_ptr);
                return std::make_unique<Raster>(gdal_ds, false);
            }),
            "Create a raster from Python GDAlDataset",
            py::arg("py_ds_ptr"))
        .def_property_readonly("width", &Raster::width)
        .def_property_readonly("length", &Raster::length)
        .def_property_readonly("shape",
                [](const Raster& self) {
                    return py::make_tuple(self.length(), self.width());
                })
        .def_property_readonly("num_bands", &Raster::numBands)
        .def_property_readonly("x0", &Raster::x0)
        .def_property_readonly("y0", &Raster::y0)
        .def_property_readonly("dx", &Raster::dx)
        .def_property_readonly("dy", &Raster::dy)
        .def_property_readonly("access", [](Raster & self)
            {
                return self.access();
            })
        .def_property_readonly("readonly", [](Raster & self)
            {
                return self.access() == 0;
            })
        .def("get_geotransform", [](Raster & self)
            {
                std::vector<double> transform(6);
                self.getGeoTransform(transform);
                return transform;
            })
        .def("set_geotransform", [](Raster & self, std::vector<double> transform)
            {
                self.setGeoTransform(transform);
            })
        .def("datatype", [](Raster & self, int i)
            {
                using T = std::underlying_type_t<GDALDataType>;
                return static_cast<T>(self.dtype(i));
            },
            py::arg("band")=1)
        .def_property_readonly("dtype",
                [](const Raster& self) {
                    ensureSingleBand(self);
                    return getRasterDType(self);
                })
        .def("get_epsg", &Raster::getEPSG)
        .def("set_epsg", &Raster::setEPSG)
        .def("get_block", &getRasterBlock,
                R"(
                Read a block of data from the raster dataset.

                Parameters
                ----------
                key : (slice, slice)
                    The row and column indices of the block to read from the
                    raster. Normal Python slicing rules apply.
                band : int, optional
                    The (1-based) index of the raster band to read from.
                    Defaults to 1.

                Returns
                -------
                numpy.ndarray
                    A 2-D NumPy array with the same datatype as the raster
                    containing the values read from the raster.
                )",
                py::arg("key"), py::arg("band") = 1)
        .def("set_block", &setRasterBlock,
                R"(
                Write a block of data to a raster.

                Parameters
                ----------
                key : (slice, slice)
                    The row and column indices of the block in the raster to be
                    overwritten. Normal Python slicing rules apply.
                value : array_like
                    The data to write to the raster.
                band : int, optional
                    The (1-based) index of the raster band to write to.
                    Defaults to 1.
                )",
                py::arg("key"), py::arg("value"), py::arg("band") = 1)
        .def("__getitem__",
                [](Raster& self, const SlicePair& key) {
                    ensureSingleBand(self);
                    return getRasterBlock(self, key);
                })
        .def("__setitem__",
                [](Raster& self, const SlicePair& key, py::buffer value) {
                    ensureSingleBand(self);
                    setRasterBlock(self, key, value);
                })
    ;

}

// end of file
