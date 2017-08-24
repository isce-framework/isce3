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

        static void cross(std::vector<double>&,std::vector<double>&,std::vector<double>&);
        static double dot(std::vector<double>&,std::vector<double>&);
        static void linComb(double,std::vector<double>&,double,std::vector<double>&,std::vector<double>&);
        static void matMat(std::vector<std::vector<double>>&,std::vector<std::vector<double>>&,std::vector<std::vector<double>>&);
        static void matVec(std::vector<std::vector<double>>&,std::vector<double>&,std::vector<double>&);
        static double norm(std::vector<double>&);
        static void tranMat(std::vector<std::vector<double>>&,std::vector<std::vector<double>>&);
        static void unitVec(std::vector<double>&,std::vector<double>&);
        static void enuBasis(double,double,std::vector<std::vector<double>>&);
    };
}}

#endif
