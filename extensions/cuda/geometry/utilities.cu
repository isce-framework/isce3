//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright: 2017-2018

#include "utilities.h"

// Utility function to get device memory
size_t isce::cuda::geometry::getDeviceMem() {
    size_t freeByte, totalByte;
    cudaMemGetInfo(&freeByte, &totalByte);
    // Round down to nearest GB
    size_t GB = 1e9;
    totalByte = (totalByte / GB) * GB;
    return totalByte;
}

// end of file
