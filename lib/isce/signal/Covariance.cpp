// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2018-
//

#include "Covariance.h"

void isce::signal::Covariance::
covariance(isce::io::Raster& HH,
                isce::io::Raster& VH,
                isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C22)
{
    isce::signal::Crossmul crsmul;
    crsmul.doppler(_doppler, _doppler);

    crsmul.prf(_prf);

    crsmul.rangeSamplingFrequency(_rangeSamplingFrequency);

    crsmul.rangeBandwidth(_rangeBandwidth);

    crsmul.wavelength(_wavelength);

    crsmul.rangePixelSpacing(_rangeSpacing);

    crsmul.rangeLooks(_rangeLooks);

    crsmul.azimuthLooks(_azimuthLooks);

    crsmul.doCommonAzimuthbandFiltering(false);

    crsmul.doCommonRangebandFiltering(false);

    crsmul.crossmul(HH, HH, C11);

    crsmul.crossmul(HH, VH, C12);

    crsmul.crossmul(VH, VH, C22);

}

void isce::signal::Covariance::
covariance(isce::io::Raster& HH,
                isce::io::Raster& VH,
                isce::io::Raster& HV,
                isce::io::Raster& VV,
                isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C13,
                isce::io::Raster& C14,
                isce::io::Raster& C22,
                isce::io::Raster& C23,
                isce::io::Raster& C24,
                isce::io::Raster& C33,
                isce::io::Raster& C34,
                isce::io::Raster& C44)
{
    isce::signal::Crossmul crsmul;
    crsmul.doppler(_doppler, _doppler);

    crsmul.prf(_prf);

    crsmul.rangeSamplingFrequency(_rangeSamplingFrequency);

    crsmul.rangeBandwidth(_rangeBandwidth);

    crsmul.wavelength(_wavelength);

    crsmul.rangePixelSpacing(_rangeSpacing);

    crsmul.rangeLooks(_rangeLooks);

    crsmul.azimuthLooks(_azimuthLooks);

    crsmul.doCommonAzimuthbandFiltering(false);

    crsmul.doCommonRangebandFiltering(false);

    crsmul.crossmul(HH, HH, C11);

    crsmul.crossmul(HH, VH, C12);

    crsmul.crossmul(HH, HV, C13); 

    crsmul.crossmul(HH, VV, C14);

    crsmul.crossmul(VH, VH, C22);

    crsmul.crossmul(VH, HV, C23);
 
    crsmul.crossmul(VH, VV, C24);

    crsmul.crossmul(HV, HV, C33);
 
    crsmul.crossmul(HV, VV, C34);

    crsmul.crossmul(VV, VV, C44);
      
}

void isce::signal::Covariance::
geocodeCovariance(isce::io::Raster& C11,
                isce::io::Raster& C12,
                isce::io::Raster& C22,
                isce::io::Raster& TCF,
                isce::io::Raster& GC11,
                isce::io::Raster& GC12,
                isce::io::Raster& GC13,
                isce::io::Raster& GC21,
                isce::io::Raster& GC22,
                isce::io::Raster& GC23,
                isce::io::Raster& GC31,
                isce::io::Raster& GC32,
                isce::io::Raster& GC33)
{
    
    // buffers for blocks of data

    //for each block in the geocoded grid:
        
        //read a block in radar coordintae for C11, C12, C22, RTC
        
        //RTC correction
        
        //Polarization estimation/correction
        
        //Effective number of looks
        
        //Faraday rotation estimation/correction
        
        //Symmetrization
                
        //Geocode 
    
}

void isce::signal::Covariance::
_rtcCorrection(std::valarray<std::complex<float>>& input, 
                std::valarray<float>& TCF)
{
    input *= TCF; 
}

void isce::signal::Covariance::
_orientationAngle(std::valarray<float>& azimuthSlope,
                std::valarray<float>& rangeSlope,
                std::valarray<float>& lookAngle,
                std::valarray<float>& tau)
{
    tau = std::atan2(std::tan(azimuthSlope), 
                    std::sin(lookAngle) - 
                        std::tan(rangeSlope)*std::cos(lookAngle));

}

void isce::signal::Covariance::
_correctOrientation(std::valarray<float>& tau, 
                    std::valarray<std::complex<float>>& C11,
                    std::valarray<std::complex<float>>& C12,
                    std::valarray<std::complex<float>>& C13,
                    std::valarray<std::complex<float>>& C21,
                    std::valarray<std::complex<float>>& C22,
                    std::valarray<std::complex<float>>& C23,
                    std::valarray<std::complex<float>>& C31,
                    std::valarray<std::complex<float>>& C32,
                    std::valarray<std::complex<float>>& C33)
{
    size_t arraySize = tau.size();
    std::valarray<float> R11(arraySize);
    std::valarray<float> R12(arraySize);
    


    R11 = 1.0 + std::cos(2*tau);
    R12 = std::sqrt(2)*std::sin(2*tau);
    R13 = 2.0 - R11;

    R21 = -1.0*R12;
    R22 = 2.0*(R11 - 1.0);
    R23 = R12;

    R31 = 2.0 - R11;
    R32 = -1*R12; 
    R33 = R11;

    // 
    c11 = 0.25*(R11*(C11*R11 + C12*R12 + C13*R13) +
                R12*(C21*R11 + C22*R12 + C23*R13) +
                R13*(C31*R11 + C32*R12 + C33*R13));

    c12 = 0.25*(R11*(C11*R21 + C12*R22 + C13*R23) + 
                R12*(C21*R21 + C22*R22 + C23*R23) + 
                R13*(C31*R21 + C32*R22 + C33*R23));

    c13 = 0.25*(R11*(C11*R31 + C12*R32 + C13*R33) +
                R12*(C21*R31 + C22*R32 + C23*R33) +
                R13*(C31*R31 + C32*R32 + C33*R33));

    c21 = 0.25*(R21*(C11*R11 + C12*R12 + C13*R13) +
                R22*(C21*R11 + C22*R12 + C23*R13) +
                R23*(C31*R11 + C32*R12 + C33*R13));

    c22 = 0.25*(R21*(C11*R21 + C12*R22 + C13*R23) +
                R22*(C21*R21 + C22*R22 + C23*R23) +
                R23*(C31*R21 + C32*R22 + C33*R23));

    c23 = 0.25*(R21*(C11*R31 + C12*R32 + C13*R33) +
                R22*(C21*R31 + C22*R32 + C23*R33) +
                R23*(C31*R31 + C32*R32 + C33*R33));

    c31 = 0.25*(R31*(C11*R11 + C12*R12 + C13*R13) +
                R32*(C21*R11 + C22*R12 + C23*R13) +
                R33*(C31*R11 + C32*R12 + C33*R13));

    c32 = 0.25*(R31*(C11*R21 + C12*R22 + C13*R23) +
                R32*(C21*R21 + C22*R22 + C23*R23) +
                R33*(C31*R21 + C32*R22 + C33*R23));

    c33 = 0.25*(R31*(C11*R31 + C12*R32 + C13*R33) +
                R32*(C21*R31 + C22*R32 + C23*R33) +
                R33*(C31*R31 + C32*R32 + C33*R33));

}

void isce::signal::Covariance::
_faradayRotationAngle(std::valarray<std::complex<float>>& Shh,
                    std::valarray<std::complex<float>>& Shv,
                    std::valarray<std::complex<float>>& Svh,
                    std::valarray<std::complex<float>>& Svv,
                    std::valarray<float>& delta)
{
    delta = 0.25*std::atan2(-2*std::real((Shv - Svh)*std::conj(Shh + Svv)) ,
                            std::pow(std::abs(Shh + Svv), 2) - 
                            std::pow(std::abs(Shv - Svh), 2) );


}

void isce::signal::Covariance::
_correctFaradayRotation()
{
    
}

void isce::signal::Covariance::
_symmetrization()
{
    
}


