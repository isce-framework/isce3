//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef ISCELIB_LINALG_H
#define ISCELIB_LINALG_H

#include <vector>

namespace isceLib {
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
        static void enuBasis(double,double,vector<vector<double>>&);
    };
}

#endif

