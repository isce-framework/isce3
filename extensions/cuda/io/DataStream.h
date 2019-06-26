#pragma once
#ifndef ISCE_CUDA_IO_DATASTREAM_H
#define ISCE_CUDA_IO_DATASTREAM_H

#include <isce/cuda/core/Event.h>
#include <isce/cuda/core/Stream.h>
#include <isce/io/Raster.h>
#include <string>
#include <thrust/host_vector.h>
#include <thrust/system/cuda/experimental/pinned_allocator.h>

namespace isce { namespace cuda { namespace io {

// thrust::host_vector whose data buffer uses page-locked memory
template<typename T>
using pinned_host_vector = thrust::host_vector<T,
        thrust::system::cuda::experimental::pinned_allocator<T>>;

// callback for asynchronously writing to files via std::ofstream
class fstreamCallback {
private:
    fstreamCallback() = default;

    fstreamCallback(const std::string * filename,
                    const char * buffer,
                    std::size_t offset,
                    std::size_t count);

    // interface for cudaStreamAddCallback
    // casts *obj* to (fstreamCallback *) and calls write()
    static
    void CUDART_CB cb_write(cudaStream_t, cudaError_t, void * obj);

    // write the contents of *buffer* to file at *filename* via std::ofstream
    void write();

    const std::string * filename;
    const char * buffer;
    std::size_t offset;
    std::size_t count;

    friend class FileDataStream;
};

// callback for asynchronously writing to Raster objects
class RasterCallback {
private:
    RasterCallback() = default;

    RasterCallback(isce::io::Raster * raster,
                   char * buffer,
                   std::size_t col,
                   std::size_t row,
                   std::size_t width,
                   std::size_t length);

    // interface for cudaStreamAddCallback
    // casts *obj* to (RasterCallback *) and calls setBlock()
    template<typename T>
    static
    void CUDART_CB cb_setBlock(cudaStream_t, cudaError_t, void * obj);

    // forwards members as arguments to Raster::setBlock()
    template<typename T>
    void setBlock();

    isce::io::Raster * raster;
    char * buffer;
    std::size_t col;
    std::size_t row;
    std::size_t width;
    std::size_t length;

    friend class RasterDataStream;
};

/**
 * Utility class for asynchronously reading/writing between files and
 * device memory.
 */
class FileDataStream {
public:
    FileDataStream() = default;

    /**
     * Constructor
     *
     * @param[in] filename path to file
     * @param[in] stream CUDA stream
     * @param[in] buffer_size stream buffer size in bytes
     */
    FileDataStream(const std::string & filename,
                   isce::cuda::core::Stream stream,
                   std::size_t buffer_size = 0);

    /** Get path to file. */
    const std::string & filename() const;

    /** Set path to file. */
    void set_filename(const std::string &);

    /** Get associated CUDA stream object. */
    isce::cuda::core::Stream stream() const;

    /** Set CUDA stream. */
    void set_stream(isce::cuda::core::Stream stream);

    /** Get stream buffer size in bytes. */
    std::size_t buffer_size() const;

    /** Set stream buffer size in bytes. */
    void resize_buffer(std::size_t buffer_size);

    /**
     * Read data and copy to the current device asynchronously.
     *
     * Reading from the file may overlap with operations in other CUDA
     * streams. Copying to the device may overlap with operations on
     * the host or in other CUDA streams.
     *
     * The destination memory address must be in device-accessible memory.
     *
     * @param[in] dst destination memory address
     * @param[in] offset position of first character in file to read
     * @param[in] count size in bytes to read
     */
    void load(void * dst, std::size_t offset, std::size_t count);

    /**
     * Write data from the current device to the file asynchronously.
     *
     * Copying from the device may overlap with operations on the host
     * or in other CUDA streams. Writing to the file may overlap with
     * operations in other CUDA streams.
     *
     * The source memory address must be in device-accessible memory.
     *
     * @param[in] src source memory address
     * @param[in] offset position to write first character in file
     * @param[in] count size in bytes to write
     */
    void store(const void * src, std::size_t offset, std::size_t count);

private:
    std::string _filename;
    isce::cuda::core::Stream _stream = 0;
    isce::cuda::core::Event _mutex;
    pinned_host_vector<char> _buffer;
    fstreamCallback _callback;
};

/**
 * Utility class for asynchronously reading/writing between Rasters and
 * device memory.
 */
class RasterDataStream {
public:
    RasterDataStream() = default;

    /**
     * Constructor
     *
     * @param[in] raster pointer to raster
     * @param[in] stream CUDA stream
     * @param[in] buffer_size stream buffer size in bytes
     */
    RasterDataStream(isce::io::Raster * raster,
                     isce::cuda::core::Stream stream,
                     std::size_t buffer_size = 0);

    /** Get pointer to Raster object. */
    isce::io::Raster * raster() const;

    /** Set raster. */
    void set_raster(isce::io::Raster *);

    /** Get associated CUDA stream object. */
    isce::cuda::core::Stream stream() const;

    /** Set CUDA stream. */
    void set_stream(isce::cuda::core::Stream stream);

    /** Get stream buffer size in bytes. */
    std::size_t buffer_size() const;

    /** Set stream buffer size in bytes. */
    void resize_buffer(std::size_t buffer_size);

    /**
     * Read a block of data from the Raster and copy to the current
     * device asynchronously.
     *
     * Reading from the Raster may overlap with operations in other CUDA
     * streams. Copying to the device may overlap with operations on
     * the host or in other CUDA streams.
     *
     * The destination memory address must be in device-accessible memory.
     *
     * @param[in] dst destination memory address
     * @param[in] col index of first column to read
     * @param[in] row index of first row to read
     * @param[in] width number of columns to read
     * @param[in] length number of rows to read
     */
    template<typename T>
    void load(T * dst,
            std::size_t col,
            std::size_t row,
            std::size_t width,
            std::size_t length);

    /**
     * Write a block of data from the current device to the
     * Raster asynchronously.
     *
     * Copying from the device may overlap with operations on the host
     * or in other CUDA streams. Writing to the Raster may overlap with
     * operations in other CUDA streams.
     *
     * The source memory address must be in device-accessible memory.
     *
     * @param[in] src source memory address
     * @param[in] col index of first column to write
     * @param[in] row index of first row to write
     * @param[in] width number of columns to write
     * @param[in] length number of rows to write
     */
    template<typename T>
    void store(const T * src,
            std::size_t col,
            std::size_t row,
            std::size_t width,
            std::size_t length);

private:
    isce::io::Raster * _raster;
    isce::cuda::core::Stream _stream = 0;
    isce::cuda::core::Event _mutex;
    pinned_host_vector<char> _buffer;
    RasterCallback _callback;
};

}}}

#define ISCE_CUDA_IO_DATASTREAM_ICC
#include "DataStream.icc"
#undef ISCE_CUDA_IO_DATASTREAM_ICC

#endif

