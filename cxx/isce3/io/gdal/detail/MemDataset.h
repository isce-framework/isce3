#pragma once

#include <string>

namespace isce3 { namespace io { namespace gdal { namespace detail {

template<typename T>
inline
std::string getMemDatasetName(T * data,
                              int width,
                              int length,
                              int bands,
                              std::size_t colstride,
                              std::size_t rowstride,
                              std::size_t bandstride);

}}}}

#define ISCE_IO_GDAL_DETAIL_MEMDATASET_ICC
#include "MemDataset.icc"
#undef ISCE_IO_GDAL_DETAIL_MEMDATASET_ICC
