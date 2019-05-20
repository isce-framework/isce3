//-*- C++ -*-
//-*- coding: utf-8 -*-
//
// Author: Bryan V. Riel, Joshua Cohen
// Copyright 2017-2018

#ifndef ISCE_CORE_BASIS_H
#define ISCE_CORE_BASIS_H

// isce::core
#include "Cartesian.h"
#include "Constants.h"

// Declaration
namespace isce {
    namespace core {
        class Basis;
    }
}

/** Simple class to store three-dimensional basis vectors*/
class isce::core::Basis {

    public:
        /** Default constructor*/
        CUDA_HOSTDEV
        Basis() {};

        /**Constructor with basis vectors*/
        CUDA_HOSTDEV
        Basis(const Vec3& x0, const Vec3& x1, const Vec3& x2) :
            _x0(x0), _x1(x1), _x2(x2) {}

        /** Geocentric TCN constructor
         * @param[in] p position vector
         * @param[in] v position vector */
        CUDA_HOSTDEV explicit Basis(const Vec3& p, const Vec3& v) {
            const Vec3 n =         -p.unitVec();
            const Vec3 c = n.cross(v).unitVec();
            const Vec3 t = c.cross(n).unitVec();
            _x0 = t;
            _x1 = c;
            _x2 = n;
        }

        /**Return first basis vector*/
        CUDA_HOSTDEV const Vec3& x0() const { return _x0; }

        /**Return second basis vector*/
        CUDA_HOSTDEV const Vec3& x1() const { return _x1; }

        /**Return third basis vector*/
        CUDA_HOSTDEV const Vec3& x2() const { return _x2; }

        /**Set the first basis vector*/
        CUDA_HOSTDEV void x0(const Vec3& x0) { _x0 = x0; }

        /**Set the second basis vector*/
        CUDA_HOSTDEV void x1(const Vec3& x1) { _x1 = x1; }

        /**Set the third basis vecot*/
        CUDA_HOSTDEV void x2(const Vec3& x2) { _x2 = x2; }

        /** \brief Project a given vector onto basis
         *
         * @param[in] vec 3D vector to project
         * @param[out] res 3D vector output
         *
         * \f[
         *      res_i = (x_i \cdot vec)
         *  \f] */
        CUDA_HOSTDEV inline Vec3 project(Vec3& vec) {
            return Vec3 { _x0.dot(vec),
                          _x1.dot(vec),
                          _x2.dot(vec) };
        };

        /** \brief Combine the basis with given weights
         *
         * @param[in] vec 3D vector to use as weights
         * @param[out] res 3D vector output
         *
         * \f[
         *      res = \sum_{i=0}^2 vec[i] \cdot x_i
         * \f] */
        inline void combine(cartesian_t &vec, cartesian_t &res) {
            for(int ii =0; ii < 3; ii++) {
                res[ii] = vec[0] * _x0[ii] + vec[1] * _x1[ii] + vec[2] * _x2[ii];
            }
        };

    private:
        cartesian_t _x0;
        cartesian_t _x1;
        cartesian_t _x2;
};

#endif

// end of file
