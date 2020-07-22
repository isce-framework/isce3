#pragma once

#include "forward.h"

#include <array>
#include <gdal_priv.h>

#include "detail/GDALDataTypeUtil.h"

namespace isce3 { namespace io { namespace gdal {

/** Interface to 2-D memory array */
class Buffer {
public:

    /**
     * Create a buffer object describing a read-only row-major, contiguous data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] datatype Datatype identifier
     * \param[in] shape Buffer dims (nrows, ncols)
     */
    constexpr
    Buffer(const void * data,
           GDALDataType datatype,
           const std::array<int, 2> & shape);

    /**
     * Create a buffer object describing a row-major, contiguous data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] datatype Datatype identifier
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] access Access mode
     */
    constexpr
    Buffer(void * data,
           GDALDataType datatype,
           const std::array<int, 2> & shape,
           GDALAccess access = GA_Update);

    /**
     * Create a buffer object describing a read-only data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] datatype Datatype identifier
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] strides Stride in bytes between the start of adjacent elements along each dimension
     */
    constexpr
    Buffer(const void * data,
           GDALDataType datatype,
           const std::array<int, 2> & shape,
           const std::array<std::size_t, 2> & strides);

    /**
     * Create a buffer object describing a data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] datatype Datatype identifier
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] strides Stride in bytes between the start of adjacent elements along each dimension
     * \param[in] access Access mode
     */
    constexpr
    Buffer(void * data,
           GDALDataType datatype,
           const std::array<int, 2> & shape,
           const std::array<std::size_t, 2> & strides,
           GDALAccess access = GA_Update);

    /** Pointer to the start of the data buffer */
    constexpr
    void * data() { return _data; }

    /** Pointer to the start of the data buffer */
    constexpr
    const void * data() const { return _data; }

    /** Datatype identifier */
    constexpr
    GDALDataType datatype() const { return _datatype; }

    /** Size in bytes of a single element */
    constexpr
    std::size_t itemsize() const { return detail::getSize(_datatype); }

    /** Buffer dims (nrows, ncols) */
    constexpr
    const std::array<int, 2> & shape() const { return _shape; }

    /** Number of rows */
    constexpr
    int length() const { return _shape[0]; }

    /** Number of columns */
    constexpr
    int width() const { return _shape[1]; }

    /** Stride in bytes between the start of adjacent elements along each dimension */
    constexpr
    const std::array<std::size_t, 2> & strides() const { return _strides; }

    /** Stride in bytes between the start of adjacent rows */
    constexpr
    std::size_t rowstride() const { return _strides[0]; }

    /** Stride in bytes between the start of adjacent columns */
    constexpr
    std::size_t colstride() const { return _strides[1]; }

    /** Access mode */
    constexpr
    GDALAccess access() const { return _access; }

    /**
     * Cast to typed buffer
     *
     * \throws isce3::except::RuntimeError if the requested type does not match
     * the underlying buffer datatype
     */
    template<typename T>
    constexpr
    TypedBuffer<T> cast() const;

private:
    void * _data;
    GDALDataType _datatype;
    std::array<int, 2> _shape;
    std::array<std::size_t, 2> _strides;
    GDALAccess _access;
};

/** Buffer with static type information */
template<typename T>
class TypedBuffer {
public:

    /**
     * Create a buffer object describing a read-only row-major, contiguous data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] shape Buffer dims (nrows, ncols)
     */
    constexpr
    TypedBuffer(const T * data,
                const std::array<int, 2> & shape);

    /**
     * Create a buffer object describing a row-major, contiguous data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] access Access mode
     */
    constexpr
    TypedBuffer(T * data,
                const std::array<int, 2> & shape,
                GDALAccess access = GA_Update);

    /**
     * Create a buffer object describing a read-only data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] strides Stride in bytes between the start of adjacent elements along each dimension
     */
    constexpr
    TypedBuffer(const T * data,
                const std::array<int, 2> & shape,
                const std::array<std::size_t, 2> & strides);

    /**
     * Create a buffer object describing a data buffer
     *
     * \param[in] data Pointer to start of data buffer
     * \param[in] datatype Datatype identifier
     * \param[in] shape Buffer dims (nrows, ncols)
     * \param[in] strides Stride in bytes between the start of adjacent elements along each dimension
     * \param[in] access Access mode
     */
    constexpr
    TypedBuffer(T * data,
                const std::array<int, 2> & shape,
                const std::array<std::size_t, 2> & strides,
                GDALAccess access = GA_Update);

    /** Pointer to the start of the data buffer */
    constexpr
    T * data() { return _data; }

    /** Pointer to the start of the data buffer */
    constexpr
    const T * data() const { return _data; }

    /** Datatype identifier */
    constexpr
    GDALDataType datatype() const { return detail::Type2GDALDataType<T>::datatype; }

    /** Size in bytes of a single element */
    constexpr
    std::size_t itemsize() const { return sizeof(T); }

    /** Buffer dims (nrows, ncols) */
    constexpr
    const std::array<int, 2> & shape() const { return _shape; }

    /** Number of rows */
    constexpr
    int length() const { return _shape[0]; }

    /** Number of columns */
    constexpr
    int width() const { return _shape[1]; }

    /** Stride in bytes between the start of adjacent elements along each dimension */
    constexpr
    const std::array<std::size_t, 2> & strides() const { return _strides; }

    /** Stride in bytes between the start of adjacent rows */
    constexpr
    std::size_t rowstride() const { return _strides[0]; }

    /** Stride in bytes between the start of adjacent columns */
    constexpr
    std::size_t colstride() const { return _strides[1]; }

    /** Access mode */
    constexpr
    GDALAccess access() const { return _access; }

private:
    T * _data;
    std::array<int, 2> _shape;
    std::array<std::size_t, 2> _strides;
    GDALAccess _access;
};

}}}

#define ISCE_IO_GDAL_BUFFER_ICC
#include "Buffer.icc"
#undef ISCE_IO_GDAL_BUFFER_ICC
