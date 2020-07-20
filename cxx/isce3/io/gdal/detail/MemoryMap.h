#pragma once

#include <cpl_virtualmem.h>
#include <gdal_priv.h>
#include <memory>

#include "../forward.h"

namespace isce { namespace io { namespace gdal { namespace detail {

class MemoryMap {
public:

    MemoryMap() : _mmap(nullptr, [](CPLVirtualMem *) {}) {}

    explicit operator bool() const { return _mmap.get(); }

    // Get pointer to start of virtual memory mapping
    void * data() { return CPLVirtualMemGetAddr(_mmap.get()); }

    // Get pointer to start of virtual memory mapping
    const void * data() const { return CPLVirtualMemGetAddr(_mmap.get()); }

    // Size in bytes of mapped region
    std::size_t size() const { return CPLVirtualMemGetSize(_mmap.get()); }

    // Access mode
    CPLVirtualMemAccessMode access() const { return CPLVirtualMemGetAccessMode(_mmap.get()); }

    // Stride in bytes between the start of adjacent columns
    std::size_t colstride() const { return _colstride; }

    // Stride in bytes between the start of adjacent rows
    std::size_t rowstride() const { return _rowstride; }

    friend class isce::io::gdal::Raster;

private:

    MemoryMap(const GDALRasterBand * raster);

    MemoryMap(GDALRasterBand * raster, GDALAccess access);

    std::shared_ptr<CPLVirtualMem> _mmap;
    std::size_t _colstride = 0;
    std::size_t _rowstride = 0;
};

}}}}
