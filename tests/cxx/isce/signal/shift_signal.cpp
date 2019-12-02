// -*- C++ -*-
// -*- coding: utf-8 -*-
//
// Author: Heresh Fattahi
// Copyright 2019-

#include <iostream>
#include <cstdio>
#include <sstream>
#include <fstream>
#include <cmath>
#include <complex>
#include <gtest/gtest.h>
#include "isce/signal/Signal.h"
#include "isce/signal/shiftSignal.h"

TEST(shiftSignal, shiftSignal)
{


    size_t width = 100;
    size_t length = 1;
    
    size_t nfft;

    // instantiate a signal object
    isce::signal::Signal<double> sigObj;
    sigObj.nextPowerOfTwo(width, nfft);

    // reserve memory for a block of data with the size of nfft
    std::valarray<std::complex<double>> slc(nfft);
    std::valarray<std::complex<double>> slcShifted(nfft);
    std::valarray<std::complex<double>> spec(nfft);

    // a simple band limited signal (a linear phase ramp)
    for (size_t i = 0; i < width; ++i) {

        double phase = 2*M_PI*i*0.001;
        slc[i] = std::complex<double> (std::cos(phase), std::sin(phase));

    }

    // setup the forward FFT plan
    sigObj.forwardRangeFFT(slc, spec, nfft, length);

    // setup the inverse FFT plan
    sigObj.inverseRangeFFT(spec, slc, nfft, length);

    // A desired shift in X direction
    double shiftX = -1.0;

    // No shift in Y direction
    double shiftY = 0.0;

    // shift the signal
    isce::signal::shiftSignal(slc, slcShifted, spec,
                  nfft, length,
                  shiftX, shiftY, sigObj);

    // max error tolerance
    double max_arg_err = 0.0;
    for (size_t i = 0 ; i < width - 1; ++i) {
        
        // the phase of the expected signal
        double phaseShifted = 2*M_PI*(i - shiftX)*0.001;

        // expected shifted signal
        std::complex<double> expectedSignal = std::complex<double> 
                            (std::cos(phaseShifted), std::sin(phaseShifted));

        // difference of the expceted shifted signal with the result of the shift
        std::complex<double> diff =  slcShifted[i] *std::conj(expectedSignal);

        // compare the phase of the difference with the max_error
        if (std::arg(diff) > max_arg_err )
             max_arg_err = std::arg(diff);
    }

    std::cout << "max_arg_err : " << max_arg_err << std::endl;
    ASSERT_LT(max_arg_err, 1.0e-10);

}

int main(int argc, char * argv[]) {
      testing::InitGoogleTest(&argc, argv);
      return RUN_ALL_TESTS();
}

