
// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2020-
//
#include "filterKernel.h"

#include <cmath>
#include <iostream>

#include <isce3/except/Error.h>

std::valarray<double> isce3::signal::boxcar2D(const int& columns,
                                              const int& rows)
{
    // container for the kernel
    std::valarray<double> kernel(columns * rows);
    // sanity checks
    if (columns <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of columns should be > 0");
    }
    if (rows <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of rows should be > 0");
    }
    if (kernel.size() != columns * rows) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Kernel's size is not consistent with input columns and rows");
    }

    double sum = columns * rows;
    kernel = 1.0 / sum;

    return kernel;
}

std::valarray<double> isce3::signal::boxcar1D(const int& length)
{

    // kernel container
    std::valarray<double> kernel(length);

    // sanity checks
    if (length <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's length should be > 0");
    }
    if (kernel.size() != length) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's size is not consistent with "
                                         "the input length parameter");
    }

    double sum = length;
    kernel = 1.0 / sum;

    return kernel;
}

std::valarray<double> isce3::signal::gaussian1D(const int& length,
                                                const double& sigma)
{

    // kernel container
    std::valarray<double> kernel(length);

    // sanity checks
    if (length <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's length should be > 0");
    }
    // sanity checks
    if (sigma <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's sigma should be > 0");
    }

    double sum = 0.0;
    for (int line = 0; line < length; line++) {
        double m = (length - 1.0) / 2.0;
        double x = line - m;
        double val = std::exp(-1.0 * (x * x / (2 * sigma * sigma)));

        sum += val;

        kernel[line] = val;
    }

    kernel /= sum;

    return kernel;
}

std::valarray<double> isce3::signal::gaussian2D(const int& columns,
                                                const int& rows,
                                                const double& sigmaX,
                                                const double& sigmaY)
{

    // kernel container
    std::valarray<double> kernel(columns * rows);

    // sanity checks
    if (columns <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of columns should be > 0");
    }
    if (rows <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of rows should be > 0");
    }
    if (sigmaX <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's sigmaX should be > 0");
    }
    if (sigmaY <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's sigmaY should be > 0");
    }

    double sum = 0.0;
    for (int line = 0; line < rows; line++) {
        double ym = (rows - 1.0) / 2.0;
        double y = line - ym;
        for (int col = 0; col < columns; col++) {
            double xm = (columns - 1.0) / 2.0;
            double x = col - xm;
            double val = std::exp(-1.0 * (x * x / (2 * sigmaX * sigmaX) +
                                          y * y / (2 * sigmaY * sigmaY)));

            sum += val;

            kernel[col + line * columns] = val;
        }
    }

    kernel /= sum;

    return kernel;
}

std::valarray<double> isce3::signal::delta2D(const int& columns,
                                             const int& rows)
{

    // kernel container
    std::valarray<double> kernel(columns * rows);

    // sanity checks
    if (columns <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of columns should be > 0");
    }
    if (rows <= 0) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(), "Kernel's number of rows should be > 0");
    }
    if (kernel.size() != columns * rows) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Kernel's size is not consistent with input columns and rows");
    }
    // Even if we allow the function runs with even window sizes,
    // we may want to give a warning message
    int center_row = rows / 2;
    int center_col = columns / 2;

    kernel = 0.0;
    kernel[center_row * columns + center_col] = 1;

    return kernel;
}

std::valarray<double> isce3::signal::delta1D(const int& length)
{
    // kernel container
    std::valarray<double> kernel(length);

    // sanity checks
    if (length <= 0) {
        throw isce3::except::DomainError(ISCE_SRCINFO(),
                                         "Kernel's length should be > 0");
    }
    if (kernel.size() != length) {
        throw isce3::except::DomainError(
                ISCE_SRCINFO(),
                "Kernel's size is not consistent with input length");
    }
    int center = length / 2;
    std::cout << "center: " << center << std::endl;
    kernel = 0.0;
    kernel[center] = 1;

    return kernel;
}
