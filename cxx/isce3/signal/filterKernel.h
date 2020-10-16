// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2020-

#pragma once

#include "forward.h"

#include <valarray>

namespace isce3 { namespace signal {

/**
 * Create a 1D Gaussian kernel.
 * \param[in] length length of the kernel
 * \param[in] sigma standard deviation of the gaussian kernel
 * @returns The output 1D Gaussian kernel
 */
std::valarray<double> gaussian1D(const int& length, const double& sigma);

/**
 * Creates a 2D Gaussian kernel.
 * \param[in] columns number of columns in the Gaussian kernel
 * \param[in] rows number of rows in the Gaussian kernel
 * \param[in] sigmaX standard deviation of the Gaussian kernel in columns
 * direction \param[in] sigmaY standard deviation of the Gaussian kernel in rows direction
 * @returns The output 2D Gaussian kernel
 */
std::valarray<double> gaussian2D(const int& columns, const int& rows,
                                 const double& sigmaX, const double& sigmaY);

/**
 * Create a 1D boxcar kernel.
 * \param[in] length length of the kernel
 * @returns The output 1D boxcar kernel
 */
std::valarray<double> boxcar1D(const int& length);

/**
 * Create a 2D boxcar kernel.
 * \param[in] columns number of columns in the kernel
 * \param[in] rows number of rows in the kernel
 * @returns output 2D boxcar kernel
 */
std::valarray<double> boxcar2D(const int& columns, const int& rows);

/**
 * Create a 1D ideal delta kernel. (one at the center and zero everywhere else)
 * \param[in] length length of the kernel
 * @returns output 1D delta kernel (out[floor(length/2)] = 1)
 */
std::valarray<double> delta1D(const int& length);

/**
 * Create a 2D ideal delta kernel. (one at the center and zero everywhere else)
 * \param[in] columns number columns in the kernel
 * \param[in] rows number rows in the kernel
 * @returns output 2D delta kernel
 */
std::valarray<double> delta2D(const int& columns, const int& rows);

}} // namespace isce3::signal
