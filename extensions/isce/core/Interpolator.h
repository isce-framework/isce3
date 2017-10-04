//
// Author: Joshua Cohen
// Copyright 2017
//

#ifndef __ISCE_CORE_INTERPOLATOR_H__
#define __ISCE_CORE_INTERPOLATOR_H__

#include <vector>

namespace isce { namespace core {
    struct Interpolator {
        Interpolator() = default;

        template<class U>
        static U bilinear(double,double,const std::vector<std::vector<U>>&);
        
        template<class U>
        static U bicubic(double,double,const std::vector<std::vector<U>>&);
        
        static void sinc_coef(double,double,int,double,int,int&,int&,std::vector<double>&);
        
        template<class U, class V>
        static U sinc_eval(const std::vector<U>&,const std::vector<V>&,int,int,int,double,int);
        
        template<class U, class V>
        static U sinc_eval_2d(const std::vector<std::vector<U>>&,const std::vector<V>&,int,int,int,
                              int,double,double,int,int);
        
        static float interp_2d_spline(int,int,int,const std::vector<std::vector<float>>&,double,
                                      double);
        static double quadInterpolate(const std::vector<double>&,const std::vector<double>&,double);
        static double akima(int,int,const std::vector<std::vector<float>>&,double,double);
    };

    void initSpline(const std::vector<double>&,int,std::vector<double>&,std::vector<double>&);
    double spline(double,const std::vector<double>&,int,const std::vector<double>&);
}}

#endif
