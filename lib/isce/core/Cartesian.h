/*
 * Compatibility definitions for legacy `cartesian_t` type
 */

#pragma once
#ifndef ISCE_CORE_CARTESIAN_H
#define ISCE_CORE_CARTESIAN_H

#include "Vector.h"
#include "DenseMatrix.h"

namespace isce { namespace core {
    typedef isce::core::Vec3 cartesian_t;
    typedef isce::core::Mat3 cartmat_t;
}}

#endif
