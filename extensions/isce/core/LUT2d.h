//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan Riel
// Copyright 2017
//

#ifndef ISCELIB_LUT2D_H
#define ISCELIB_LUT2D_H

#include <cstddef>
#include <vector>
//#include "Raster.h"

namespace isce { namespace core {

    template <typename T>
    class LUT2d {

        // Convenience
        typedef std::size_t size_t;

        public:

            // Vectors to hold indices in both dimensions
            std::vector<double> x_index;
            std::vector<double> y_index;

            // 2D vector to hold values
            std::vector<std::vector<T>> values;

            // Basic constructor does nothing
            LUT2d() {};
            // Constructor from vectors of indices and values
            LUT2d(std::vector<double> &, std::vector<double> &,
                  std::vector<std::vector<T>> &);

            // Destructor
            ~LUT2d();

            // Methods 
            void setDimensions();
            void reset();
            T eval(double, double);
            //void setValuesFromRaster(Raster<T> &);

        private:

            // Sizes of each dimension
            size_t _xsize;
            size_t _ysize;

    }; // class LUT2d

} // namespace core
} // namespace isce

#endif
