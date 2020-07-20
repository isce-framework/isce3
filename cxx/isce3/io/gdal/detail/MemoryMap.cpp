#include "MemoryMap.h"

#include <isce3/except/Error.h>

namespace isce3 { namespace io { namespace gdal { namespace detail {

MemoryMap::MemoryMap(const GDALRasterBand * raster)
:
    MemoryMap(const_cast<GDALRasterBand *>(raster), GA_ReadOnly)
{}

MemoryMap::MemoryMap(GDALRasterBand * raster, GDALAccess access)
:
    _mmap(nullptr, [](CPLVirtualMem *) {})
{
    GDALRWFlag rwflag = (access == GA_ReadOnly) ? GF_Read : GF_Write;

    int colstride;
    GIntBig rowstride;
    CPLVirtualMem * mmap = raster->GetVirtualMemAuto(rwflag, &colstride, &rowstride, nullptr);
    if (!mmap) {
        throw isce3::except::RuntimeError(ISCE_SRCINFO(), "failed to memory map specified raster");
    }

    _mmap = std::shared_ptr<CPLVirtualMem>(mmap, [](CPLVirtualMem * mmap) { CPLVirtualMemFree(mmap); });

    _colstride = std::size_t(colstride);
    _rowstride = std::size_t(rowstride);
}

}}}}
