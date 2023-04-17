/**
 * @file  cuArrays.h
 * @brief Header file for cuArrays class
 *
 * A class describes a batch of images (in 2d arrays).
 * Each image has size (height, width)
 * The number of images (countH, countW) or (1, count).
 **/

// code guard
#ifndef __CUARRAYS_H
#define __CUARRAYS_H

#include "float2.h"

#include <iostream>
#include <fstream>
#include <cstdlib>
#include <ctime>


namespace isce3::matchtemplate::pycuampcor {

template <typename T>
class cuArrays{

public:
    int height; ///< x, row, down, length, azimuth, along the track
    int width;  // y, col, across, range, along the sight
    int size;   // one image size, height*width
    int countH; // number of images along height direction
    int countW; // number of images along width direction
    int count;  // countW*countH, number of images
    T* devData; // pointer to data in device (gpu) memory
    T* hostData; // pointer to data in host (cpu) memory

    bool is_allocated; // whether the data is allocated in device memory
    bool is_allocatedHost; // whether the data is allocated in host memory

    // default constructor, empty
    cuArrays() : height(0), width(0), size(0), countH(0), countW(0), count(0),
        devData(0), hostData(0),
        is_allocated(0), is_allocatedHost(0) {}

    // constructor for single image
    cuArrays(size_t h, size_t w) : height(h), width(w), countH(1), countW(1), count(1),
        devData(0), hostData(0),
        is_allocated(0), is_allocatedHost(0)
    {
        size = w*h;
    }

    // constructor for multiple images with a total count
    cuArrays(size_t h, size_t w, size_t n) : height(h), width(w), countH(1), countW(n), count(n),
        devData(0), hostData(0),
        is_allocated(0), is_allocatedHost(0)
    {
        size = w*h;
    }

    // constructor for multiple images with (countH, countW)
    cuArrays(size_t h, size_t w, size_t ch, size_t cw) : height(h), width(w), countH(ch), countW(cw),
        devData(0), hostData(0),
        is_allocated(0), is_allocatedHost(0)
    {
        size = w*h;
        count = countH*countW;
    }

    // memory allocation
    void allocate();
    void allocateHost();
    void deallocate();
    void deallocateHost();

    // copy data between device and host memories
    void copyToHost();
    void copyToDevice();

    // get the total size
    size_t getSize()
    {
        return size*count;
    }

    // get the total size in byte
    inline long getByteSize()
    {
        return width*height*count*sizeof(T);
    }

    // destructor
    ~cuArrays()
    {
        if(is_allocated)
            deallocate();
        if(is_allocatedHost)
            deallocateHost();
    }

    // set zeroes
    void setZero();
    // output when debugging
    void debuginfo() ;
    void debuginfo(float factor);
    // write to files
    void outputToFile(std::string fn);
    void outputHostToFile(std::string fn);

};

} // namespace

#endif //__CUARRAYS_H
//end of file
