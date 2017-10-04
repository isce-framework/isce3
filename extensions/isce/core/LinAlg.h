//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_LINALG_H__
#define __ISCE_CORE_LINALG_H__

#include <vector>

namespace isce { namespace core {
    struct LinAlg {
        LinAlg() = default;

        static void cross(const std::vector<double>&,const std::vector<double>&,
                          std::vector<double>&);
        static double dot(const std::vector<double>&,const std::vector<double>&);
        static void linComb(double,const std::vector<double>&,double,const std::vector<double>&,
                            std::vector<double>&);
        static void matMat(const std::vector<std::vector<double>>&,
                           const std::vector<std::vector<double>>&,
                           std::vector<std::vector<double>>&);
        static void matVec(const std::vector<std::vector<double>>&,const std::vector<double>&,
                           std::vector<double>&);
        static double norm(const std::vector<double>&);
        static void tranMat(const std::vector<std::vector<double>>&,
                            std::vector<std::vector<double>>&);
        static void unitVec(const std::vector<double>&,std::vector<double>&);
        static void enuBasis(double,double,std::vector<std::vector<double>>&);
    };
}}

#endif
